# Sử dụng base image của Python
FROM python:3.9-slim

# Đặt biến môi trường để đảm bảo Python không lưu cache .pyc
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Cập nhật hệ thống và cài đặt các dependencies cần thiết
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tạo và đặt thư mục làm việc
WORKDIR /app

# Copy file requirements.txt vào image
COPY requirements.txt /app/

# Cài đặt các thư viện Python từ file requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào image
COPY . /app

# Mở cổng cần thiết (thay đổi nếu ứng dụng của bạn sử dụng cổng khác)
EXPOSE 5000

# Đặt lệnh chạy ứng dụng của bạn
CMD ["python", "app.py"]
