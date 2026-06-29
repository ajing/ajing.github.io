import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const sitemapPaths = [
  path.resolve(__dirname, '../dist/sitemap.xml'),
  path.resolve(__dirname, '../dist/priority-sitemap.xml'),
];

try {
  const existingSitemaps = sitemapPaths.filter(fs.existsSync);

  if (existingSitemaps.length === 0) {
    console.log('No sitemap files found to clean.');
  }

  for (const sitemapPath of existingSitemaps) {
    const originalContent = fs.readFileSync(sitemapPath, 'utf8');

    // Replace <urlset ...> with a clean tag containing only standard sitemap namespace
    const cleanContent = originalContent.replace(
      /<urlset\s+[^>]*>/i,
      '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    );

    if (originalContent !== cleanContent) {
      fs.writeFileSync(sitemapPath, cleanContent, 'utf8');
      console.log(`Successfully cleaned namespaces from ${path.basename(sitemapPath)}.`);
    }
  }
} catch (error) {
  console.error('Error cleaning sitemap namespaces:', error);
}
