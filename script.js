// Variables globales para almacenar elementos del DOM y estado de la aplicación
let video, startBtn, stopBtn, status, cameraOff, results, prediction, confidence;
let stream = null;           // Stream de la cámara
let model = null;            // Modelo de TensorFlow.js cargado
let detectionInterval = null; // Intervalo para detección automática

// Configuración de la aplicación
const MODEL_PATH = './frutas_cnn_savedmodel_tfjs/model.json'; // Ruta al modelo
const INPUT_SIZE = { width: 320, height: 258, channels: 3 };   // Tamaño de entrada del modelo
// Lista de frutas que puede detectar el modelo (en orden específico)
const FRUIT_CLASSES = [
    'Manzana', 'Plátano', 'Carambola', 'Guayaba', 'Kiwi', 'Mango', 
    'Naranja', 'Durazno', 'Pera', 'Caqui', 'Pitaya', 'Ciruela', 
    'Granada', 'Tomate', 'Melón'
];

// Funciones de utilidad

// Muestra mensajes de estado en la interfaz
function showStatus(message, type = 'info') {
    if (!status) return; // Si no existe el elemento status, no hace nada
    status.className = `status ${type}`; // Cambia la clase CSS para el color
    status.textContent = message;        // Actualiza el texto del mensaje
}

// Habilita/deshabilita botones según el estado de la aplicación
function updateButtons(cameraActive = false, modelLoaded = false) {
    if (startBtn) startBtn.disabled = cameraActive;  // Deshabilita "Iniciar" si cámara está activa
    if (stopBtn) stopBtn.disabled = !cameraActive;   // Deshabilita "Detener" si cámara no está activa
}

// Carga del modelo de inteligencia artificial
async function loadModel() {
    try {
        // Muestra mensaje de carga
        showStatus('Cargando modelo...', 'info');
        console.log('Cargando modelo desde:', MODEL_PATH);
        
        // Carga el modelo desde el archivo JSON
        model = await tf.loadGraphModel(MODEL_PATH);
        
        // Confirma que se cargó correctamente
        console.log('Modelo cargado exitosamente');
        showStatus('Modelo cargado', 'success');
        updateButtons(stream !== null, true); // Actualiza estado de botones
        
        return true; // Retorna éxito
    } catch (error) {
        // Si hay error, lo muestra y retorna falso
        console.error('Error cargando modelo:', error);
        showStatus('Error cargando modelo', 'error');
        return false;
    }
}

// Gestión de cámara

// Inicia la cámara del dispositivo
async function startCamera() {
    try {
        showStatus('Iniciando camara...', 'info');
        
        // Configuración de la cámara
        const constraints = {
            audio: false,                          // No necesita audio
            video: {
                width: { ideal: 640 },             // Ancho preferido
                height: { ideal: 480 },            // Alto preferido
                facingMode: { ideal: 'environment' } // Cámara trasera si está disponible
            }
        };
        
        // Solicita acceso a la cámara
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream; // Conecta el stream al elemento video
        
        // Espera a que el video esté listo
        await new Promise(resolve => {
            video.onloadedmetadata = resolve;
        });
        
        // Muestra el video y oculta el mensaje de cámara inactiva
        video.style.display = 'block';
        cameraOff.style.display = 'none';
        
        // Actualiza el estado de los botones
        updateButtons(true, model !== null);
        showStatus('Camara activa', 'success');
        
        // Si el modelo ya está cargado, inicia la detección automática
        if (model) {
            startDetection();
        }
        
    } catch (error) {
        console.error('Error iniciando camara:', error);
        showStatus('Error iniciando camara', 'error');
    }
}

// Detiene la cámara y limpia todo
function stopCamera() {
    if (stream) {
        // Detiene la detección automática
        stopDetection();
        
        // Detiene todos los tracks de video
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        
        // Oculta el video y muestra el mensaje de cámara inactiva
        video.style.display = 'none';
        video.srcObject = null;
        cameraOff.style.display = 'flex';
        
        // Actualiza botones y estado
        updateButtons(false, model !== null);
        showStatus('Camara detenida', 'info');
        
        // Limpia los resultados mostrados
        prediction.textContent = '-';
        confidence.textContent = '-';
    }
}

// Procesamiento de imagen para el modelo

// Convierte la imagen del video al formato que necesita el modelo
function preprocessImage(videoElement) {
    return tf.tidy(() => { // tf.tidy() limpia automáticamente tensores temporales
        // Convierte la imagen del video a tensor (matriz numérica)
        let tensor = tf.browser.fromPixels(videoElement, INPUT_SIZE.channels);
        
        // Redimensiona la imagen al tamaño que espera el modelo (320x258)
        tensor = tf.image.resizeBilinear(tensor, [INPUT_SIZE.width, INPUT_SIZE.height]);
        
        // Normaliza los valores de píxeles de 0-255 a 0-1
        tensor = tensor.div(255.0);
        
        // Agrega una dimensión extra para el "batch" (lote de imágenes)
        tensor = tensor.expandDims(0);
        
        return tensor; // Retorna el tensor listo para el modelo
    });
}

// Función principal de predicción

// Analiza la imagen actual del video y predice qué fruta es
async function predictFruit() {
    // Verifica que todo esté listo antes de hacer predicción
    if (!model || !stream || !video) return;
    
    // Verifica que el video tenga contenido válido
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) return;
    
    try {
        // Prepara la imagen para el modelo
        const inputTensor = preprocessImage(video);
        
        // Ejecuta el modelo con la imagen
        const pred = model.predict(inputTensor);
        
        // Obtiene los datos numéricos de la predicción
        let predictionData;
        if (Array.isArray(pred)) {
            // Si retorna array, toma el primer elemento
            predictionData = await pred[0].data();
            pred.forEach(tensor => tensor.dispose()); // Limpia memoria
        } else {
            // Si es un solo tensor
            predictionData = await pred.data();
            pred.dispose(); // Limpia memoria
        }
        
        // Limpia el tensor de entrada
        inputTensor.dispose();
        
        // Convierte los resultados a formato legible
        const results = Array.from(predictionData)
            .map((confidence, index) => ({
                fruit: FRUIT_CLASSES[index], // Nombre de la fruta
                confidence: confidence        // Nivel de confianza (0-1)
            }))
            .sort((a, b) => b.confidence - a.confidence); // Ordena por confianza
        
        // Toma el resultado con mayor confianza
        const best = results[0];
        const percentage = (best.confidence * 100).toFixed(1); // Convierte a porcentaje
        
        // Muestra el resultado en la interfaz
        prediction.textContent = best.fruit;
        confidence.textContent = `${percentage}%`;
        
        // Si la confianza es alta (>50%), muestra mensaje de éxito
        if (best.confidence > 0.5) {
            showStatus(`Detectado: ${best.fruit}`, 'success');
        }
        
    } catch (error) {
        console.error('Error en prediccion:', error);
    }
}

// Sistema de detección automática

// Inicia la detección continua cada segundo
function startDetection() {
    if (detectionInterval) return; // Si ya está ejecutándose, no hace nada
    
    // Ejecuta predictFruit() cada 1000ms (1 segundo)
    detectionInterval = setInterval(async () => {
        await predictFruit();
    }, 1000);
}

// Detiene la detección automática
function stopDetection() {
    if (detectionInterval) {
        clearInterval(detectionInterval); // Cancela el intervalo
        detectionInterval = null;         // Resetea la variable
    }
}

// Función principal de inicialización

// Configura toda la aplicación cuando la página se carga
async function init() {
    try {
        // Obtiene referencias a todos los elementos HTML necesarios
        video = document.getElementById('video');           // Elemento video para mostrar cámara
        startBtn = document.getElementById('startBtn');     // Botón "Iniciar"
        stopBtn = document.getElementById('stopBtn');       // Botón "Detener" 
        status = document.getElementById('status');         // Área de mensajes de estado
        cameraOff = document.getElementById('cameraOff');   // Mensaje cuando cámara está inactiva
        results = document.getElementById('results');       // Contenedor de resultados
        prediction = document.getElementById('prediction'); // Texto del nombre de fruta
        confidence = document.getElementById('confidence'); // Texto del porcentaje de confianza
        
        // Verifica que los elementos existen
        if (!video) throw new Error('Elementos DOM no encontrados');
        
        // Configura los eventos de los botones
        if (startBtn) startBtn.addEventListener('click', startCamera); // Al hacer clic en "Iniciar"
        if (stopBtn) stopBtn.addEventListener('click', stopCamera);   // Al hacer clic en "Detener"
        
        // Verifica que el navegador soporte cámara
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error('Camara no soportada');
        }
        
        // Inicializa TensorFlow.js (librería de inteligencia artificial)
        await tf.ready();
        
        // Carga el modelo de clasificación de frutas
        await loadModel();
        
        // Configura el estado inicial de los botones
        updateButtons(false, true);
        showStatus('Aplicacion lista', 'success');
        
    } catch (error) {
        // Si hay algún error en la inicialización, lo muestra
        console.error('Error inicializando:', error);
        showStatus(`Error: ${error.message}`, 'error');
    }
}

// Ejecuta la inicialización cuando la página termine de cargar
if (document.readyState === 'loading') {
    // Si la página aún se está cargando, espera al evento DOMContentLoaded
    document.addEventListener('DOMContentLoaded', init);
} else {
    // Si ya terminó de cargar, ejecuta inmediatamente
    init();
}