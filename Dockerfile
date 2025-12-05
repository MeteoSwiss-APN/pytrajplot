FROM dockerhub.apps.cp.meteoswiss.ch/mch/python-3.13:latest AS builder
ARG VERSION
LABEL ch.meteoswiss.project=pytrajplot-${VERSION}

COPY poetry.lock pyproject.toml /src/app-root/

WORKDIR /src/app-root

RUN poetry export -o requirements.txt \
    && poetry export --with dev -o requirements_dev.txt


FROM dockerhub.apps.cp.meteoswiss.ch/mch/python-3.13:latest-slim AS base
ARG VERSION
LABEL ch.meteoswiss.project=pytrajplot-${VERSION}

COPY --from=builder /src/app-root/requirements.txt /src/app-root/requirements.txt

WORKDIR /src/app-root

RUN pip install -r requirements.txt --no-cache-dir --no-deps --root-user-action=ignore
COPY pytrajplot /src/app-root/pytrajplot
COPY pyproject.toml README.rst /src/app-root/
RUN pip install /src/app-root --no-cache-dir --root-user-action=ignore

FROM base AS tester
ARG VERSION
LABEL ch.meteoswiss.project=pytrajplot-${VERSION}

COPY --from=builder /src/app-root/requirements_dev.txt /src/app-root/requirements_dev.txt
RUN pip install -r /src/app-root/requirements_dev.txt --no-cache-dir --no-deps --root-user-action=ignore

COPY test /src/app-root/test

FROM base AS runner
ARG VERSION
LABEL ch.meteoswiss.project=pytrajplot-${VERSION}

ENV VERSION=$VERSION

# For running outside of OpenShift, we want to make sure that the container is run without root privileges
# uid 1001 is defined in the base-container-images for this purpose
USER 1001

ENTRYPOINT ["python", "-m", "pytrajplot"]
CMD []
