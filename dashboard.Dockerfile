FROM nginx:1.28-alpine
ENV DASHBOARD_UI_MODE=full
COPY config/nginx.conf /etc/nginx/conf.d/default.conf
COPY dashboard /usr/share/nginx/html
COPY docs/reference /usr/share/nginx/html/documentation
COPY config/40-dashboard-config.sh /docker-entrypoint.d/40-dashboard-config.sh
RUN chmod +x /docker-entrypoint.d/40-dashboard-config.sh
