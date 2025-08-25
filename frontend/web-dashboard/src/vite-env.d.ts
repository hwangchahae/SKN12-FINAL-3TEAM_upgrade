/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_SOCKET_URL: string
  readonly VITE_TENANT_SLUG: string
  readonly VITE_APP_NAME: string
  readonly VITE_APP_VERSION: string
  readonly VITE_APP_DESCRIPTION: string
  readonly VITE_NODE_ENV: string
  readonly VITE_ENABLE_NOTIFICATIONS: string
  readonly VITE_ENABLE_DARK_MODE: string
  readonly VITE_ENABLE_ANALYTICS: string
  readonly VITE_POLLING_INTERVAL: string
  readonly VITE_SOCKET_RECONNECT_DELAY: string
  readonly VITE_MAX_FILE_SIZE: string
  readonly VITE_ALLOWED_FILE_TYPES: string
  readonly VITE_ITEMS_PER_PAGE: string
  readonly VITE_TOAST_DURATION: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}