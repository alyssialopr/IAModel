let canvas, ctx, isDrawing = false;
let session = null;

function initCanvas() {
    canvas = document.getElementById('drawing-canvas');
    ctx = canvas.getContext('2d');
    
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 15;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
}

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;
    ctx.beginPath();
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(
        e.type === 'touchstart' ? 'mousedown' : 
        e.type === 'touchmove' ? 'mousemove' : 'mouseup', 
        {
            clientX: touch.clientX,
            clientY: touch.clientY
        }
    );
    canvas.dispatchEvent(mouseEvent);
}

function clearCanvas() {
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('prediction').textContent = 'Dessinez un chiffre pour commencer';
    document.getElementById('result').textContent = 'Résultat : ';
}

function preprocessImage() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.fillStyle = '#fff';
    tempCtx.fillRect(0, 0, 28, 28);
    tempCtx.imageSmoothingEnabled = true;
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    const input = new Float32Array(28 * 28);
    
    for (let i = 0; i < 28 * 28; i++) {
        const pixelIndex = i * 4;
        const grayscale = 0.299 * data[pixelIndex] + 0.587 * data[pixelIndex + 1] + 0.114 * data[pixelIndex + 2];
        input[i] = 1.0 - (grayscale / 255.0);
    }
    
    return input;
}

async function predict() {
    if (!session) {
        document.getElementById('result').textContent = 'Résultat : Modèle non chargé';
        document.getElementById('prediction').textContent = 'Erreur : Le modèle n\'est pas encore chargé';
        return;
    }

    try {
        const inputData = preprocessImage();
        
        const inputTensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
        
        const inputName = session.inputNames[0];
        const results = await session.run({ [inputName]: inputTensor });
        
        const outputName = session.outputNames[0];
        const outputData = results[outputName].data;
        
        const probabilities = softmax(Array.from(outputData));
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        const confidence = (probabilities[maxIndex] * 100).toFixed(1);
        
        document.getElementById('result').innerHTML = `Résultat : <strong>${maxIndex}</strong>`;
        document.getElementById('prediction').innerHTML = 
            `Le chiffre prédit est <strong>${maxIndex}</strong> avec ${confidence}% de confiance`;
        
    } catch (error) {
        console.error('Erreur de prédiction:', error);
        document.getElementById('result').textContent = 'Résultat : Erreur';
        document.getElementById('prediction').textContent = 'Erreur lors de la prédiction';
    }
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(x => x / sum);
}

async function loadModel() {
    try {
        console.log('Chargement du modèle ONNX...');
        document.getElementById('prediction').textContent = 'Chargement du modèle...';
        
        const modelUrl = 'model.onnx';
        session = await ort.InferenceSession.create(modelUrl);
        
        console.log('Modèle chargé avec succès');

        document.getElementById('prediction').textContent = 'Dessinez un chiffre pour commencer';
        document.getElementById('predict-button').disabled = false;
        
    } catch (error) {
        console.error('Erreur de chargement du modèle:', error);
        document.getElementById('prediction').textContent = 'Erreur : Impossible de charger le modèle';
        document.getElementById('predict-button').disabled = true;
    }
}

window.onload = function() {
    initCanvas();
    loadModel();
    
    document.getElementById('clear-button').addEventListener('click', clearCanvas);
};