(function () {
    // Edit this block to add photos, captions, or change the first photo.
    const profilePhotoConfig = {
        firstPhotoId: 'school',
        photos: [
        {
            id: 'school',
            src: '/assets/me/me_school.jpeg',
            location: 'Nankai University',
            note: 'My Phd Life.',
        },
        {
            id: 'megvii',
            src: '/assets/me/me_megvii.jpg',
            location: 'Beijing',
            note: 'With Yuzhi during my internship at Megvii.',
        },
        {
            id: 'sanjose',
            src: '/assets/me/me_sanjose.jpg',
            location: 'San Jose',
            note: 'With Simon during my internship at Adobe Research.',
        },
        {
            id: 'santaclara',
            src: '/assets/me/me_santaclara.JPG',
            location: 'Santa Clara',
        },
        {
            id: 'shenzhen',
            src: '/assets/me/me_shenzhen.JPG',
            location: 'Shenzhen',
        },
        {
            id: 'jeju',
            src: '/assets/me/me_jeju.JPG',
            location: 'Jeju',
        },
        {
            id: 'xiamen',
            src: '/assets/me/me_xiamen.jpg',
            location: 'Xiamen',
        },
        {
            id: 'lasvegas',
            src: '/assets/me/me_lasvegas.jpg',
            location: 'Las Vegas',
        },
        {
            id: 'beijing',
            src: '/assets/me/me_beijing.JPG',
            location: 'Beijing',
        },
        {
            id: 'tianjin',
            src: '/assets/me/me_tianjin.JPG',
            location: 'Tianjin',
        },
        {
            id: 'vancouver',
            src: '/assets/me/me_vancouver.jpg',
            location: 'Vancouver',
            note: 'When I was there for NeurIPS.',
        },
        {
            id: 'shanghai',
            src: '/assets/me/me_shanghai.jpg',
            location: 'Shanghai',
        },
        {
            id: 'nashville',
            src: '/assets/me/me_nashville.JPG',
            location: 'Nashville',
            note: 'When I was there for CVPR.',
        },
        {
            id: 'cats',
            src: '/assets/me/me_cats.JPG',
            location: 'Home',
            note: 'My cats.',
        },
        {
            id: 'hangzhou',
            src: '/assets/me/me_hangzhou.JPG',
            location: 'Hangzhou',
            note: 'With part of the Z-Image team.',
        },
        ],
    };

    function shuffle(items) {
        const shuffled = items.slice();
        for (let index = shuffled.length - 1; index > 0; index -= 1) {
            const randomIndex = Math.floor(Math.random() * (index + 1));
            [shuffled[index], shuffled[randomIndex]] = [shuffled[randomIndex], shuffled[index]];
        }
        return shuffled;
    }

    function initProfilePhotos() {
        const profilePhoto = document.getElementById('profilePhoto');
        const profilePhotos = profilePhotoConfig.photos;
        if (!profilePhoto || profilePhotos.length === 0) return;

        const locationLabel = profilePhoto.querySelector('.profile-photo-location');
        const noteLabel = profilePhoto.querySelector('.profile-photo-note');
        const defaultPhoto = profilePhotos.find((photo) => photo.id === profilePhotoConfig.firstPhotoId) || profilePhotos[0];
        const randomPhotos = profilePhotos;
        let currentPhotoId = defaultPhoto.id;
        let randomQueue = shuffle(randomPhotos);

        function setPhoto(photo) {
            currentPhotoId = photo.id;
            profilePhoto.style.backgroundImage = `url("${photo.src}")`;
            profilePhoto.dataset.photoId = photo.id;

            if (locationLabel) {
                locationLabel.textContent = photo.location;
            }
            if (noteLabel) {
                noteLabel.textContent = photo.note || '';
            }

            const ariaNote = photo.note ? ` ${photo.note}` : '';
            profilePhoto.setAttribute('aria-label', `${photo.location} profile photo.${ariaNote} Click to switch photo.`);
        }

        function getNextPhoto() {
            if (randomQueue.length === 0) {
                randomQueue = shuffle(randomPhotos);
            }
            if (randomQueue.length > 1 && randomQueue[0].id === currentPhotoId) {
                randomQueue.push(randomQueue.shift());
            }
            return randomQueue.shift();
        }

        profilePhotos.forEach((photo) => {
            const image = new Image();
            image.src = photo.src;
        });

        setPhoto(defaultPhoto);
        profilePhoto.addEventListener('click', () => {
            const nextPhoto = getNextPhoto();
            if (nextPhoto) {
                setPhoto(nextPhoto);
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initProfilePhotos);
    } else {
        initProfilePhotos();
    }
}());
