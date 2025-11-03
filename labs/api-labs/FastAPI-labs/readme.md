# FastAPI Lab 
This project provides a simple example setup for building RESTful APIs with Python.

## Features

- Fast and async web framework (FastAPI + Uvicorn)
- Built-in interactive API docs (Swagger / ReDoc)

## Installation

Clone the repo and install dependencies:

```bash
cd labs/api-labs/fastapi-labs/
python -m venv venv
source venv\Scripts\activate
pip install -r requirements.txt
```

## Run the application

```bash
cd labs/api-labs/fastapi-labs/src
python train.py
uvicorn main:app --reload
```

Visit : http://127.0.0.1:8000 for the Swagger API docs