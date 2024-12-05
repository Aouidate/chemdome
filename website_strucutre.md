chemdome/
│
├── chemdome/            # Project-level configuration (settings, urls, etc.)
│   ├── __init__.py
│   ├── settings.py      # Project settings
│   ├── urls.py          # Project-level URLs
│   ├── wsgi.py
│   └── asgi.py
│
├── landing/             # Landing page app
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py          # App-specific URLs (if needed)
│   ├── templates/
│   │   └── landing/
│   │       └── index.html
│   └── static/
│       └── landing/
│           └── style.css
│
├── pca_domain/             # PCA-related domain app
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── templates/
│   │   └── pca_domain/
│   │       └── index.html
│   └── static/
│       └── pca_domain/
│           └── style.css
│
├── leverage_domain/        # Leverage Applicability domain app
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── templates/
│   │   └── leverage_domain/
│   │       └── index.html
│   └── static/
│       └── leverage_domain/
│           └── style.css
|
├── sali/        # SALI app
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── templates/
│   │   └── sali/
│   │       └── index.html
│   └── static/
│       └── sali/
│           └── style.css
│
├── templates/           # Global templates, such as base.html
│   └── base.html        # Base template for all apps to inherit
│
└── manage.py
