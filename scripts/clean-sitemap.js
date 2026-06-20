import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const sitemapPath = path.resolve(__dirname, '../dist/sitemap-0.xml');

try {
  if (fs.existsSync(sitemapPath)) {
    const originalContent = fs.readFileSync(sitemapPath, 'utf8');
    
    // Replace <urlset ...> with a clean tag containing only standard sitemap namespace
    const cleanContent = originalContent.replace(
      /<urlset\s+[^>]*>/i,
      '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    );
    
    if (originalContent !== cleanContent) {
      fs.writeFileSync(sitemapPath, cleanContent, 'utf8');
      console.log('Successfully cleaned namespaces from sitemap-0.xml.');
    } else {
      console.log('No cleaning needed or already clean.');
    }
  } else {
    console.error(`Sitemap not found at: ${sitemapPath}`);
  }
} catch (error) {
  console.error('Error cleaning sitemap namespaces:', error);
}
