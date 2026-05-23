(function () {
    const COMMENTS_CONFIG_URL = '/comments.json';
    const systemPreference = window.matchMedia('(prefers-color-scheme: dark)');
    let themeToggle = null;
    let commentToggle = null;
    let commentsConfig = null;
    let commentsLoaded = false;
    let commentsLoading = false;
    let manualTheme = null;

    function getSystemTheme() {
        return systemPreference.matches ? 'dark' : 'light';
    }

    function ensureFloatingActions() {
        let actions = document.getElementById('floatingActions');

        if (!actions && document.body) {
            actions = document.createElement('div');
            actions.className = 'floating-actions';
            actions.id = 'floatingActions';
            document.body.appendChild(actions);
        }

        return actions;
    }

    function ensureThemeToggle() {
        const actions = ensureFloatingActions();
        let toggle = document.getElementById('themeToggle');

        if (!toggle && document.body) {
            toggle = document.createElement('button');
            toggle.className = 'theme-toggle';
            toggle.id = 'themeToggle';
            toggle.type = 'button';
            toggle.dataset.label = 'Theme';
        }

        if (toggle && actions && toggle.parentElement !== actions) {
            actions.appendChild(toggle);
        }

        return toggle;
    }

    function getPagePath() {
        let pathname = window.location.pathname || '/';
        pathname = pathname.replace(/\/index\.html$/, '/');
        return pathname || '/';
    }

    function getBlogSlug() {
        const params = new URLSearchParams(window.location.search);
        return params.get('post') || params.get('slug') || document.body.dataset.blogPost || '';
    }

    function getDisqusIdentifier() {
        const path = getPagePath();

        if (path === '/blog/post.html') {
            const slug = getBlogSlug();
            return slug ? `blog:${slug}` : 'blog:post';
        }

        return `page:${path}`;
    }

    function getDisqusUrl() {
        const siteUrl = String(commentsConfig.siteUrl || window.location.origin).replace(/\/$/, '');
        return `${siteUrl}${getPagePath()}${window.location.search || ''}`;
    }

    function ensureCommentPanel() {
        let panel = document.getElementById('commentPanel');

        if (!panel && document.body) {
            panel = document.createElement('section');
            panel.className = 'comment-panel';
            panel.id = 'commentPanel';
            panel.setAttribute('aria-label', 'Comments');
            panel.setAttribute('aria-hidden', 'true');
            panel.innerHTML = `
                <div class="comment-panel-header">
                    <div>
                        <div class="comment-panel-title">Comments</div>
                        <div class="comment-panel-subtitle">Powered by Disqus</div>
                    </div>
                    <button class="comment-panel-close" id="commentPanelClose" type="button" aria-label="Close comments">x</button>
                </div>
                <div class="comment-panel-body">
                    <div class="comment-panel-status" id="commentPanelStatus">Loading comments...</div>
                    <div id="disqus_thread"></div>
                </div>`;
            document.body.appendChild(panel);

            const closeButton = document.getElementById('commentPanelClose');
            if (closeButton) {
                closeButton.addEventListener('click', closeCommentPanel);
            }
        }

        return panel;
    }

    function openCommentPanel() {
        const panel = ensureCommentPanel();
        if (!panel || !commentsConfig) return;

        panel.classList.add('open');
        panel.setAttribute('aria-hidden', 'false');
        if (commentToggle) {
            commentToggle.setAttribute('aria-expanded', 'true');
        }

        loadComments();
    }

    function closeCommentPanel() {
        const panel = document.getElementById('commentPanel');
        if (!panel) return;

        panel.classList.remove('open');
        panel.setAttribute('aria-hidden', 'true');
        if (commentToggle) {
            commentToggle.setAttribute('aria-expanded', 'false');
        }
    }

    function toggleCommentPanel() {
        const panel = ensureCommentPanel();
        if (!panel) return;

        if (panel.classList.contains('open')) {
            closeCommentPanel();
        } else {
            openCommentPanel();
        }
    }

    function setCommentStatus(message, isError) {
        const status = document.getElementById('commentPanelStatus');
        if (!status) return;

        status.textContent = message;
        status.classList.toggle('error', Boolean(isError));
    }

    function loadComments() {
        if (commentsLoaded || commentsLoading || !commentsConfig || !commentsConfig.shortname) return;

        commentsLoading = true;
        setCommentStatus('Loading comments...', false);

        window.disqus_config = function () {
            this.page.url = getDisqusUrl();
            this.page.identifier = getDisqusIdentifier();
            this.page.title = document.title;
        };

        const script = document.createElement('script');
        script.src = `https://${commentsConfig.shortname}.disqus.com/embed.js`;
        script.setAttribute('data-timestamp', String(Date.now()));
        script.async = true;
        script.onload = function () {
            commentsLoaded = true;
            commentsLoading = false;
            setCommentStatus('', false);
        };
        script.onerror = function () {
            commentsLoading = false;
            setCommentStatus('Could not load Disqus. Check the shortname and trusted domain settings.', true);
        };
        document.body.appendChild(script);
    }

    function ensureCommentToggle() {
        const actions = ensureFloatingActions();
        if (!actions || !commentsConfig || !commentsConfig.enabled || !commentsConfig.shortname) return null;

        let toggle = document.getElementById('commentToggle');

        if (!toggle) {
            toggle = document.createElement('button');
            toggle.className = 'comment-toggle';
            toggle.id = 'commentToggle';
            toggle.type = 'button';
            toggle.dataset.label = 'Comments';
            toggle.setAttribute('aria-label', 'Open comments');
            toggle.setAttribute('aria-expanded', 'false');
            toggle.innerHTML = '<span class="comment-toggle-icon" aria-hidden="true">D</span>';
            toggle.addEventListener('click', toggleCommentPanel);
        }

        if (toggle.parentElement !== actions) {
            actions.insertBefore(toggle, actions.firstChild);
        }

        return toggle;
    }

    async function initComments() {
        try {
            const response = await fetch(COMMENTS_CONFIG_URL);
            if (!response.ok) return;

            const config = await response.json();
            if (!config || !config.enabled || !config.shortname) return;

            commentsConfig = {
                enabled: Boolean(config.enabled),
                shortname: String(config.shortname).trim(),
                siteUrl: String(config.siteUrl || '').trim()
            };

            commentToggle = ensureCommentToggle();
        } catch (error) {}
    }

    function applyTheme(theme) {
        document.documentElement.dataset.theme = theme;

        themeToggle = ensureThemeToggle();
        if (!themeToggle) return;

        const isDark = theme === 'dark';
        themeToggle.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
        themeToggle.dataset.label = isDark ? 'Light' : 'Dark';
        themeToggle.innerHTML = `<span class="theme-toggle-icon" aria-hidden="true">${isDark ? '&#9728;' : '&#9790;'}</span>`;
    }

    function applyCurrentTheme() {
        applyTheme(manualTheme || getSystemTheme());
    }

    function initTheme() {
        themeToggle = ensureThemeToggle();
        if (themeToggle) {
            themeToggle.addEventListener('click', function () {
                const currentTheme = document.documentElement.dataset.theme || getSystemTheme();
                manualTheme = currentTheme === 'dark' ? 'light' : 'dark';
                applyTheme(manualTheme);
            });
        }

        applyCurrentTheme();
        initComments();
    }

    if (systemPreference.addEventListener) {
        systemPreference.addEventListener('change', function () {
            manualTheme = null;
            applyCurrentTheme();
        });
    } else if (systemPreference.addListener) {
        systemPreference.addListener(function () {
            manualTheme = null;
            applyCurrentTheme();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTheme);
    } else {
        initTheme();
    }
}());
