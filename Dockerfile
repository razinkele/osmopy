# Stage 1: Java runtime base
FROM eclipse-temurin:17-jre AS java-base

# Stage 2: Python application with Java copied in
FROM python:3.12-slim

# Copy Java runtime from stage 1
ENV JAVA_HOME=/opt/java/openjdk
COPY --from=java-base $JAVA_HOME $JAVA_HOME
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# System dependencies for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash osmose

WORKDIR /app

# Copy all application code
COPY pyproject.toml .
COPY osmose/ osmose/
COPY ui/ ui/
COPY data/ data/
COPY app.py .

# Copy OSMOSE Java JAR (wildcard avoids failure if directory is empty/missing)
COPY osmose-java* osmose-java/

# Install Python dependencies
RUN pip install --no-cache-dir .

RUN chown -R osmose:osmose /app
USER osmose

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
