# Render image for the isometric scenes environment (README, "Setup").
# Microsoft's official Playwright image holds Python, the browsers (in /ms-playwright) and fonts, but not
# the Playwright Python package, which is installed here at the version that matches those browsers.
# Build from the environment folder: prime images push <name>:<tag> --context . --dockerfile render_image/Dockerfile
FROM mcr.microsoft.com/playwright/python:v1.52.0-noble@sha256:a1d2b48b65f41f34e5e1d7690f385f1376397b7208691cba359e4078edbf86cb
RUN python3 -m pip install --no-cache-dir --break-system-packages playwright==1.52.0
COPY svg_scoring/render_frames.py /opt/render/render_frames.py
