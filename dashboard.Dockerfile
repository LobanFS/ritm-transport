FROM nginx:1.28-alpine
ENV DASHBOARD_UI_MODE=full RITM_IMPORT_MAX_BYTES=268435456 RITM_IMPORT_TIMEOUT_SECONDS=120
COPY config/nginx.conf /etc/nginx/templates/default.conf.template
COPY dashboard /usr/share/nginx/html
COPY docs/reference /usr/share/nginx/html/documentation
COPY config/40-dashboard-config.sh /docker-entrypoint.d/40-dashboard-config.sh
RUN chmod +x /docker-entrypoint.d/40-dashboard-config.sh
