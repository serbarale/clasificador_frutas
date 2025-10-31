// Variables globales
let video = document.getElementById('video');
let stream = null;
let model = null;
let isDetecting = false;
let detectionInterval = null;

// Elementos
const startBtn = document.getElementById('startBtn');
const detectBtn = document.getElementById('detectBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');
const cameraOff = document.getElementById('cameraOff');
const results = document.getElementById('results');
const prediction = document.getElementById('prediction');
const confidence = document.getElementById('confidence');

// Clases de frutas
const fruitClasses = [
    'Apple', 'Banana', 'Carambola', 'Guava', 'Kiwi', 'Mango', 
    'Orange', 'Peach', 'Pear', 'Persimmon', 'Pitaya', 'Plum', 
    'Pomegranate', 'Tomatoes', 'Muskmelon'
];

// Canvas secundario para procesamiento del modelo
const modelCanvas = document.getElementById('modelCanvas');
const modelCtx = modelCanvas.getContext('2d');

// Funciones de utilidad
function showStatus(message, type = 'info') {
    status.className = `status ${type}`;
    status.textContent = message;
    status.style.display = 'block';
    
    setTimeout(() => {
        status.style.display = 'none';
    }, 3000);
}

function updateButtons(cameraActive) {
    startBtn.disabled = cameraActive;
    detectBtn.disabled = !cameraActive || !model;
    stopBtn.disabled = !cameraActive;
}

// Iniciar cámara
async function startCamera() {
    try {
        showStatus('Accediendo a la cámara...', 'info');
        
        stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        video.srcObject = stream;
        video.style.display = 'block';
        cameraOff.style.display = 'none';
        
        showStatus('Cámara iniciada', 'success');
        updateButtons(true);
        
        // Iniciar detección automática si el modelo está cargado
        if (model) {
            setTimeout(() => {
                startDetection();
                showStatus('🔍 Detección automática activada', 'info');
            }, 1000);
        }

    } catch (error) {
        console.error('Error:', error);
        showStatus('Error al acceder a la cámara', 'error');
        updateButtons(false);
    }
}

// Detener cámara
function stopCamera() {
    if (stream) {
        // Detener detección
        stopDetection();
        
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        
        video.srcObject = null;
        video.style.display = 'none';
        cameraOff.style.display = 'block';
        
        showStatus('Cámara detenida', 'info');
        updateButtons(false);
    }
}

// Cargar modelo TensorFlow.js
async function loadModel() {
    try {
        showStatus('Cargando modelo de IA...', 'info');
        
        // Cargar el modelo desde la carpeta_salida
        model = await tf.loadLayersModel('./carpeta_salida/model.json');
        
        console.log('Modelo cargado exitosamente');
        console.log('Input shape:', model.inputs[0].shape);
        console.log('Output shape:', model.outputs[0].shape);
        
        showStatus('Modelo de IA cargado correctamente', 'success');
        
        // Actualizar botones si la cámara ya está activa
        if (stream) {
            updateButtons(true);
        }
        
    } catch (error) {
        console.error('Error cargando el modelo:', error);
        showStatus('Error al cargar el modelo de IA', 'error');
        model = null;
    }
}

// Procesar imagen para el modelo (320x258)
function preprocessImageForModel(videoElement) {
    return tf.tidy(() => {
        // Dibujar el video actual en el canvas oculto con el tamaño requerido
        modelCtx.drawImage(videoElement, 0, 0, 320, 258);
        
        // Convertir canvas a tensor
        let tensor = tf.browser.fromPixels(modelCanvas);
        
        // Normalizar valores de 0-255 a 0-1
        tensor = tensor.div(255.0);
        
        // Agregar dimensión de lote [1, 320, 258, 3]
        tensor = tensor.expandDims(0);
        
        return tensor;
    });
}

// Realizar predicción
async function predictFruit(showInResults = false) {
    if (!model || !stream) return;
    
    try {
        showStatus('🔍 Analizando...', 'info');
        
        // Preprocesar imagen
        const preprocessed = preprocessImageForModel(video);
        
        // Realizar predicción
        const predictions = await model.predict(preprocessed).data();
        
        // Encontrar la clase con mayor probabilidad
        const maxIndex = predictions.indexOf(Math.max(...predictions));
        const confidenceValue = predictions[maxIndex];
        const predictedFruit = fruitClasses[maxIndex];
        
        // Obtener top 3 predicciones
        const top3 = Array.from(predictions)
            .map((conf, index) => ({ fruit: fruitClasses[index], confidence: conf }))
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 3);
        
        if (showInResults) {
            // Mostrar en área de resultados detallada
            results.style.display = 'block';
            
            if (confidenceValue > 0.3) {
                const percentage = (confidenceValue * 100).toFixed(1);
                prediction.innerHTML = `<strong>${predictedFruit}</strong>`;
                
                let confidenceHtml = `<strong>Confianza: ${percentage}%</strong><br><br>`;
                confidenceHtml += '<strong>Top 3 predicciones:</strong><br>';
                top3.forEach((item, index) => {
                    const percent = (item.confidence * 100).toFixed(1);
                    confidenceHtml += `${index + 1}. ${item.fruit}: ${percent}%<br>`;
                });
                confidence.innerHTML = confidenceHtml;
                
                showStatus(`✅ Detectado: ${predictedFruit}`, 'success');
            } else {
                prediction.innerHTML = '<strong>No detectado</strong>';
                confidence.innerHTML = 'Confianza muy baja. Intenta acercar una fruta a la cámara.';
                showStatus('❌ No se pudo detectar fruta', 'error');
            }
        } else {
            // Mostrar solo en status (modo automático)
            if (confidenceValue > 0.6) {
                const percentage = (confidenceValue * 100).toFixed(1);
                showStatus(`🍎 ${predictedFruit} (${percentage}%)`, 'success');
            }
        }
        
        // Limpiar tensor de memoria
        preprocessed.dispose();
        
    } catch (error) {
        console.error('Error en predicción:', error);
        showStatus('Error al analizar imagen', 'error');
    }
}

// Iniciar detección en tiempo real
function startDetection() {
    if (!model || isDetecting) return;
    
    isDetecting = true;
    
    // Realizar predicción cada 2 segundos
    detectionInterval = setInterval(() => {
        if (stream && model) {
            predictFruit();
        }
    }, 2000);
    
    console.log('Detección iniciada');
}

// Detener detección
function stopDetection() {
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
    isDetecting = false;
    console.log('Detección detenida');
}

// Inicialización cuando el DOM esté listo
function initializeApp() {
    // Obtener elementos del DOM
    video = document.getElementById('video');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const status = document.getElementById('status');
    const cameraOff = document.getElementById('cameraOff');
    
    // Event listeners
    startBtn.addEventListener('click', startCamera);
    detectBtn.addEventListener('click', () => predictFruit(true));
    stopBtn.addEventListener('click', stopCamera);

    // Verificar soporte
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showStatus('Tu navegador no soporta acceso a la cámara', 'error');
        startBtn.disabled = true;
    }

    // Cargar modelo
    loadModel();
}

// Esperar a que el DOM esté completamente cargado
document.addEventListener('DOMContentLoaded', initializeApp);