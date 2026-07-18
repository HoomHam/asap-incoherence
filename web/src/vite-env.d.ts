/// <reference types="vite/client" />

declare module './wasm/kasap.js' {
  const createModule: (opts?: object) => Promise<unknown>;
  export default createModule;
}

declare module '*.bin?url' {
  const url: string;
  export default url;
}

declare module '*.wasm?url' {
  const url: string;
  export default url;
}

declare module 'plotly.js-dist-min';

declare module '*.png' {
  const url: string;
  export default url;
}
