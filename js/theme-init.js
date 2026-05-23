(function () {
    try {
        localStorage.removeItem('site-theme');
    } catch (error) {}

    document.documentElement.removeAttribute('data-theme');
}());
