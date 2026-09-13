/**
 * Browser shim for onnxruntime-web when loaded via CDN <script> tag (window.ort).
 */
const getOrt = (): any => {
  if (typeof globalThis !== 'undefined' && (globalThis as any).ort) {
    return (globalThis as any).ort;
  }
  return {};
};

const ortProxy: any = new Proxy(
  {},
  {
    get(_target, prop) {
      const ort = getOrt();
      return ort[prop];
    },
  },
);

export default ortProxy;
export const InferenceSession = new Proxy(
  {},
  {
    get(_target, prop) {
      return getOrt().InferenceSession?.[prop];
    },
  },
);
export const Tensor: any = function (this: any, ...args: any[]) {
  const OrtTensor = getOrt().Tensor;
  return new OrtTensor(...args);
};
export const env: any = new Proxy(
  {},
  {
    get(_target, prop) {
      return getOrt().env?.[prop];
    },
    set(_target, prop, value) {
      const ort = getOrt();
      if (ort.env) {
        ort.env[prop] = value;
      }
      return true;
    },
  },
);
