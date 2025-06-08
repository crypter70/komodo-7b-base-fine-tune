FROM python:3.11

# change dir
WORKDIR /code

# add req.txt
COPY ./requirements.txt /code/requirements.txt

# installation
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# running api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]