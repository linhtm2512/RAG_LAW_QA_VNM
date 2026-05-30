## Local Development

To run the backend server locally, follow these steps:

1. Install `tesseract-ocr` and the Vietnamese language pack:

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-vie
```

2. Fill in the environment variables by copying the example file:

```bash
cp .env.example .env
```

3. Install dependencies in a virtual environment:

```bash
source ./.venv/bin/activate
pip install -r requirements.txt
```

4. Run the backend server:

```bash
python main.py
```

With `uv`:

```bash
uv sync
uv run main.py
```
