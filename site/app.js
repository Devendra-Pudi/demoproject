const root = document.getElementById('cards');
const repository = 'https://github.com/Devendra-Pudi/demoproject/tree/arena/01a0c7ca-demoproject/projects/';
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}
fetch('projects.json').then(response => {
  if (!response.ok) throw new Error('Directory unavailable');
  return response.json();
}).then(projects => {
  root.replaceChildren();
  for (const project of projects) {
    const card = element('article', '', 'card');
    const top = element('div', '', 'card-top');
    top.append(element('span', project.number, 'number'), element('span', project.stack));
    const actions = element('div', '', 'card-actions');
    const source = element('a', 'View project ↗');
    source.href = repository + project.folder;
    actions.append(source);
    if (project.url && new URL(project.url).protocol === 'https:') {
      const launch = element('a', 'Open application ↗');
      launch.href = project.url;
      launch.rel = 'noopener noreferrer';
      actions.append(launch);
    } else {
      actions.append(element('span', 'Deployment URL not configured', 'status'));
    }
    card.append(top, element('h3', project.name), element('p', project.description),
                element('p', project.detail, 'detail'), actions);
    root.append(card);
  }
}).catch(() => {
  root.replaceChildren(element('p', 'Unable to load the directory. Please use the GitHub source link above.'));
});
