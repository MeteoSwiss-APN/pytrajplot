# doc/_static

This directory is the place for Sphinx "static" assets (CSS, JavaScript, images, fonts) that should be copied verbatim into the built documentation.

Why this directory exists
- Sphinx (by default) copies the files in `doc/_static` into the final HTML output.
- Git does not track empty directories, so we keep this README here to ensure the directory is present in the repo and to document how to add files.

Quick steps to add a custom CSS file

1. Create the file under this directory, for example:

   doc/_static/custom.css

   Example minimal CSS (save as `doc/_static/custom.css`):

   ```css
   /* Example: doc/_static/custom.css */
   :root { --brand-color: #005ea6; }
   body { font-family: "Helvetica Neue", Arial, sans-serif; }
   h1, h2, h3 { color: var(--brand-color); }
   ```

2. Tell Sphinx to include the CSS. Open `doc/conf.py` and ensure `html_static_path` contains `'_static'` (this is usually the default):

   ```py
   html_static_path = ['_static']
   ```

   Then add the CSS to be included in the HTML. There are two common ways:

   - Preferred (modern Sphinx versions): add the file name to `html_css_files`:
     ```py
     html_css_files = ['custom.css']
     ```

   - Or, in `conf.py` provide a `setup` function to register the file (useful if you need conditional logic):
     ```py
     def setup(app):
         # For Sphinx >= 1.8
         app.add_css_file('custom.css')

         # If you must support older Sphinx (<= 1.7) you can use:
         # app.add_stylesheet('custom.css')
     ```

   After this, build your docs (e.g. `make -C doc html` or `sphinx-build -b html doc doc/_build/html`) and the CSS will be included in the generated pages.

Git notes: why adding this README matters
- Git doesn't track empty directories. If `doc/_static` has no files, it won't be added to the repository.
- Keep this `README.md` (or add a `.gitkeep` file) so the directory is present in the repo.

To add the README and any static file to git:

```bash
# create dir if needed
mkdir -p doc/_static
# create files (either custom CSS or a placeholder .gitkeep)
# e.g. touch doc/_static/.gitkeep
# or add your custom.css and this README

# stage and commit
git add doc/_static/README.md doc/_static/custom.css
git commit -m "Add doc/_static and instructions for custom static files"
```

Best practices and tips
- Use meaningful names (e.g. `custom.css`, `theme-overrides.css`).
- Keep only files needed for the docs in `_static`.
- If you have many static assets or subfolders, keep paths relative and list only the top-level `_static` in `html_static_path` (Sphinx copies the entire contents).
- If you need different CSS per builder or environment, register it from `setup(app)` with conditional logic.
