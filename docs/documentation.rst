Building this documentation
===========================

The API documentation is built with Sphinx from the docstrings under
``lanfear/``, plus the hand-written guide pages in ``docs/``. On every push to
``main``, ``.github/workflows/docs.yml`` renders it to Markdown and pushes it to
this repo's `wiki <https://github.com/alex-rawlings/lanfear/wiki>`_, one page per
``.rst`` file. ``Home.md`` (and ``_Sidebar.md``, if present) is hand-maintained
on GitHub and never touched by this workflow, so link any new page from it there.

To build it locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html   # -> docs/_build/html/index.html (multi-page; for local browsing)
