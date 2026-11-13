import './styles/app.css';
import { GolYaPoochApp } from './app/appController';

const root = document.getElementById('app');
if (!root) {
  throw new Error('Root #app element not found');
}

const app = new GolYaPoochApp(root);
app.mount();
