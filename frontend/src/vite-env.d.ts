/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend base URL for a split deploy (e.g. the Render URL). Empty in dev. */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
