export const withTrailingSlash = (path: string) => {
  const [pathname, suffix = ""] = path.split(/([?#].*)/, 2);

  if (
    !pathname ||
    pathname === "/" ||
    pathname.endsWith("/") ||
    /\/[^/]+\.[^/]+$/.test(pathname)
  ) {
    return path;
  }

  return `${pathname}/${suffix}`;
};
