// Injects a button to capture the current frame of the Reel video
(function() {
    function addCaptureButton() {
        if (document.getElementById('ig-capture-frame-btn')) return;
        const btn = document.createElement('button');
        btn.id = 'ig-capture-frame-btn';
    btn.innerText = 'Capture Frame & Show Video URL';
        btn.style.position = 'fixed';
        btn.style.top = '20px';
        btn.style.right = '20px';
        btn.style.zIndex = 9999;
        btn.style.padding = '10px 16px';
        btn.style.background = '#fff';
        btn.style.border = '2px solid #3897f0';
        btn.style.borderRadius = '8px';
        btn.style.color = '#3897f0';
        btn.style.fontWeight = 'bold';
        btn.style.cursor = 'pointer';
        btn.onclick = function() {
            const video = document.querySelector('video');
            if (!video) {
                alert('No video found!');
                return;
            }
            // Get the live video URL
            const videoUrl = video.src;
            if (videoUrl) {
                alert('Video URL: ' + videoUrl);
            } else {
                alert('Video URL not found!');
            }
            // Capture the current frame as before
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/png');
            window.open(chrome.runtime.getURL('cropper.html') + '?img=' + encodeURIComponent(dataUrl), '_blank');
        };
        document.body.appendChild(btn);
    }
    setInterval(addCaptureButton, 2000);
})();
