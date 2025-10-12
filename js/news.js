document.addEventListener('DOMContentLoaded', function() {
    const newsContainer = document.querySelector('.news-container');
    const newsToggle = document.getElementById('news-toggle');

    if (newsToggle) {
        newsToggle.addEventListener('click', function(e) {
            e.preventDefault();
            newsContainer.classList.toggle('expanded');
            if (newsContainer.classList.contains('expanded')) {
                newsToggle.textContent = 'collapse';
            } else {
                newsToggle.textContent = 'expand';
                newsContainer.scrollTop = 0; // Reset scroll position
            }
        });
    }
});
