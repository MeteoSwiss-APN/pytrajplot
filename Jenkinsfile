class Globals {
    // set to true to abort the pipeline if the SonarQube quality gate fails
    static boolean qualityGateAbortPipeline = false

    // Threshold for mypy issues before failing the build
    static int mypyIssueThreshold = 10

    // Name of the container image
    static String containerImageName= ''

    // Semantic version of the artifact
    static String semanticVersion = ''

    // Pin mchbuild to stable version to avoid breaking changes
    static String mchbuildPipPackage = 'mchbuild>=0.11.3,<0.12.0'
}

String rebuild_cron = env.BRANCH_NAME == "main" ? "@midnight" : ""

pipeline {
    agent { label 'podman' }

    triggers { cron(rebuild_cron) }

    options {
        // Prevent concurrent builds; new builds wait for the current one to finish
        disableConcurrentBuilds()

        // Automatically discard old builds and artifacts to save space
        buildDiscarder(logRotator(
            artifactDaysToKeepStr: '7',  // Keep Jenkins artifacts for 7 days
            artifactNumToKeepStr: '1',  // Keep only the latest Jenkins artifact
            daysToKeepStr: '45',  // Keep build records for 45 days
            numToKeepStr: '10'  // Keep the last 10 builds
        ))

        // Set a timeout for the pipeline build (1 hour)
        timeout(time: 1, unit: 'HOURS')

        // Use the specified GitLab connection for SCM integration to publish pipeline status
        gitLabConnection('CollabGitLab')
    }

    environment {
        PATH = "$workspace/.venv-mchbuild/bin:$HOME/tools/openshift-client-tools:$PATH"
        HTTP_PROXY = 'http://proxy.meteoswiss.ch:8080'
        HTTPS_PROXY = 'http://proxy.meteoswiss.ch:8080'
        NO_PROXY = '.meteoswiss.ch,localhost'
        SCANNER_HOME = tool name: 'Sonarqube-certs-PROD', type: 'hudson.plugins.sonar.SonarRunnerInstallation'
    }

    stages {
        stage('Preflight') {
            steps {
                updateGitlabCommitStatus name: 'Build', state: 'running'

                script {
                    echo '---- INSTALLING MCHBUILD ----'
                    sh """
                        python -m venv .venv-mchbuild
                        PIP_INDEX_URL=https://hub.meteoswiss.ch/nexus/repository/python-all/simple \
                            .venv-mchbuild/bin/pip install --upgrade "${Globals.mchbuildPipPackage}"
                    """

                    echo '---- INITIALIZING PARAMETERS ----'
                    if (env.TAG_NAME) {
                        echo "Release build detected, triggered by tag: ${env.TAG_NAME}."
                        def isMajorMinorPatch = sh(
                            script: "mchbuild -s version=${env.TAG_NAME} -g isMajorMinorPatch build.checkGivenSemanticVersion",
                            returnStdout: true
                        )
                        if (isMajorMinorPatch != 'true') {
                            currentBuild.result = 'ABORTED'
                            error("The provided tag '${env.TAG_NAME}' does not follow the semantic versioning" +
                                " format <major>.<minor>.<patch>. Aborting release.")
                        }
                        Globals.semanticVersion = env.TAG_NAME
                    } else
                    {
                        echo "Development build detected, triggered from a branch."
                        Globals.semanticVersion= sh(
                            script: 'mchbuild -g semanticVersion build.getSemanticVersion',
                            returnStdout: true
                        )
                    }

                    Globals.containerImageName = sh(
                        script: 'mchbuild -g containerImageName build.getImageName',
                        returnStdout: true
                    )
                    echo "Using semantic version: ${Globals.semanticVersion}"
                    echo "Using container image name: ${Globals.containerImageName}"
                }
            }
        }

        stage('Build') {
            steps {
                echo '---- BUILDING CONTAINER IMAGES ----'
                sh """
                    mchbuild -s semanticVersion=${Globals.semanticVersion} -s containerImageName=${Globals.containerImageName} build.artifacts
                """

                echo("---- RUNNING UNIT TESTS & COLLECTING COVERAGE ----")
                sh """
                    mchbuild -s semanticVersion=${Globals.semanticVersion} -s containerImageName=${Globals.containerImageName} test.unit
                """
            }
            post {
                always {
                    junit keepLongStdio: true, testResults: 'test_reports/junit*.xml'
                }
            }
        }

        stage('Scan') {
            steps {
                echo '---- LINTING & TYPE CHECKING ----'
                sh """
                    mchbuild -s semanticVersion=${Globals.semanticVersion} -s containerImageName=${Globals.containerImageName} test.lint
                """

                script {
                    // Mypy quality gate
                    def annotatedReport = scanForIssues(
                        tool: myPy(pattern: 'test_reports/mypy.log'),
                    )
                    publishIssues issues: [annotatedReport]
                    def totalMypyIssues = annotatedReport.size()
                    if (totalMypyIssues > Globals.mypyIssueThreshold) {
                        error("Too many mypy issues detected (${totalMypyIssues} > ${Globals.mypyIssueThreshold}). Aborting build.")
                    }
                }

                echo("---- SONARQUBE ANALYSIS ----")
                withSonarQubeEnv("Sonarqube-PROD") {
                    // Adjust source paths in coverage.xml for compatibility with SonarQube
                    // This is necessary due to differences in file structure when using Podman
                    // Reference: https://stackoverflow.com/questions/57220171/sonarqube-client-fails-to-parse-pytest-coverage-results
                    sh "sed -i 's/\\/src\\/app-root/.\\//g' test_reports/coverage.xml"
                    sh "${SCANNER_HOME}/bin/sonar-scanner"
                }

                echo("---- SONARQUBE QUALITY GATE ----")
                timeout(time: 1, unit: 'HOURS') {
                    // If the quality gate fails, the pipeline will be aborted based on the configured flag
                    waitForQualityGate abortPipeline: Globals.qualityGateAbortPipeline
                }
            }
        }

        stage('Publish Artifacts & Docs') {
            environment {
                REGISTRY_AUTH_FILE = "$workspace/.containers/auth.json"
            }
            steps {
                echo "---- PUBLISHING CONTAINER IMAGES ----"
                withCredentials([usernamePassword(credentialsId: 'openshift-nexus', passwordVariable: 'NXPASS', usernameVariable: 'NXUSER')]) {
                    sh """
                        mchbuild -s semanticVersion=${Globals.semanticVersion} -s containerImageName=${Globals.containerImageName} publish.artifacts
                    """
                }
            }
        }
    }

    post {
        cleanup {
            echo '---- CLEANING UP WORKSPACE ----'
            sh """
                mchbuild -s semanticVersion=${Globals.semanticVersion} clean
            """
        }
        aborted {
            echo 'Build was aborted.'
            updateGitlabCommitStatus name: 'Build', state: 'canceled'
        }
        failure {
            echo 'Build failed. Sending notification email...'
            updateGitlabCommitStatus name: 'Build', state: 'failed'
            sh 'df -h'
            emailext(subject: "${currentBuild.fullDisplayName}: ${currentBuild.currentResult}",
                attachLog: true,
                attachmentsPattern: 'generatedFile.txt',
                to: env.BRANCH_NAME == 'main' ?
                    sh(script: "mchbuild -g notifyOnNightlyFailure", returnStdout: true) : '',
                body: "Job '${env.JOB_NAME} #${env.BUILD_NUMBER}': ${env.BUILD_URL}",
                recipientProviders: [requestor(), developers()])
        }
        success {
            echo 'Build completed successfully.'
            updateGitlabCommitStatus name: 'Build', state: 'success'
        }
    }
}
