## Local Development

To run the backend server locally, follow these steps:

Install `tesseract-ocr` and the Vietnamese language pack:

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-vie
```

Fill in the environment variables by copying the example file:

```bash
cp .env.example .env
```

Install dependencies in a virtual environment:

```bash
source ./.venv/bin/activate
pip install -r requirements.txt
```

Run the backend server:

```bash
python main.py
```
