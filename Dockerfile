# Stage 1: Java runtime base
FROM eclipse-temurin:17-jre AS java-base

# Stage 2: the C# engine (spec docs/superpowers/specs/2026-11-17-csharp-engine-design.md, 9.7),
# published from the SDK image. The project embeds osmose/engine_cs/features.json.
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS csharp-build
WORKDIR /src
COPY csharp/ csharp/
COPY osmose/engine_cs/features.json osmose/engine_cs/features.json
RUN DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 \
    dotnet publish csharp/src/Osmose.Engine/Osmose.Engine.csproj -c Release -o /out

# Stage 3: the .NET runtime alone, copied in below the way JAVA_HOME is
FROM mcr.microsoft.com/dotnet/runtime:10.0 AS dotnet-runtime

# Stage 4: Python application with Java and .NET copied in
FROM python:3.12-slim

# Copy Java runtime from stage 1
ENV JAVA_HOME=/opt/java/openjdk
COPY --from=java-base $JAVA_HOME $JAVA_HOME
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# The C# engine (optional; the app runs without it). The assembly is built with
# InvariantGlobalization, so the slim image needs no ICU. OSMOSE_CSHARP_ENGINE_DIR points the
# package at the published assembly wherever osmose is imported from.
ENV DOTNET_ROOT=/opt/dotnet
COPY --from=dotnet-runtime /usr/share/dotnet $DOTNET_ROOT
COPY --from=csharp-build /out /opt/osmose-csharp
ENV OSMOSE_CSHARP_ENGINE_DIR=/opt/osmose-csharp

# System dependencies for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libstdc++6 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash osmose

WORKDIR /app

# Copy all application code
COPY pyproject.toml .
COPY osmose/ osmose/
COPY ui/ ui/
COPY data/ data/
COPY www/ www/
COPY app.py .

# osmose is pip-installed into site-packages, so demo.py's data/-sibling lookup
# misses the /app/data copy above — point it at the bundled data explicitly.
ENV OSMOSE_DATA_DIR=/app/data

# Copy OSMOSE Java JAR (wildcard avoids failure if directory is empty/missing)
COPY osmose-java* osmose-java/

# Install Python dependencies (csharp: pythonnet, the bridge to the C# engine)
RUN pip install --no-cache-dir ".[csharp]"

RUN chown -R osmose:osmose /app
USER osmose

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
