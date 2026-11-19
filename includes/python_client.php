<?php
declare(strict_types=1);

const PROJECT_ROOT = __DIR__ . DIRECTORY_SEPARATOR . '..';

/**
 * Locate the Python executable inside the project's virtual environment.
 */
function find_python_executable(): string
{
    $root = realpath(PROJECT_ROOT) ?: PROJECT_ROOT;
    $candidates = [
        $root . DIRECTORY_SEPARATOR . '.venv' . DIRECTORY_SEPARATOR . 'Scripts' . DIRECTORY_SEPARATOR . 'python.exe',
        $root . DIRECTORY_SEPARATOR . '.venv' . DIRECTORY_SEPARATOR . 'bin' . DIRECTORY_SEPARATOR . 'python',
    ];
    foreach ($candidates as $candidate) {
        if (is_file($candidate)) {
            return $candidate;
        }
    }
    return 'python';
}

/**
 * Try sending the uploaded photo to the persistent local Python Flask API.
 * Returns null if the Flask server is not running or does not serve /api/predict.
 *
 * @return array{status: int, data?: array<string, mixed>, error?: string}|null
 */
function call_python_http_api(string $tmpPath, string $originalName): ?array
{
    if (!function_exists('curl_init')) {
        return null;
    }

    $apiUrl = getenv('PYTHON_BACKEND_URL');
    if (!is_string($apiUrl) || trim($apiUrl) === '') {
        $apiUrl = 'http://127.0.0.1:5000/api/predict';
    }

    $mime = 'application/octet-stream';
    if (function_exists('finfo_open')) {
        $finfo = finfo_open(FILEINFO_MIME_TYPE);
        if ($finfo !== false) {
            $detected = finfo_file($finfo, $tmpPath);
            if (is_string($detected) && $detected !== '') {
                $mime = $detected;
            }
            finfo_close($finfo);
        }
    }

    $ch = curl_init($apiUrl);
    if ($ch === false) {
        return null;
    }

    $postFields = [
        'file' => new CURLFile($tmpPath, $mime, $originalName !== '' ? $originalName : 'upload.jpg'),
    ];

    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => $postFields,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_CONNECTTIMEOUT_MS => 350,
        CURLOPT_TIMEOUT => 30,
        CURLOPT_HTTPHEADER => ['Accept: application/json'],
    ]);

    $responseBody = curl_exec($ch);
    $curlErrNo = curl_errno($ch);
    $httpCode = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($curlErrNo !== 0 || !is_string($responseBody) || $responseBody === '') {
        return null;
    }

    $decoded = json_decode($responseBody, true);
    if (!is_array($decoded)) {
        return null;
    }

    if ($httpCode === 200 && !empty($decoded['ok'])) {
        return ['status' => 200, 'data' => $decoded];
    }

    if ($httpCode === 400 && isset($decoded['error'])) {
        return ['status' => 400, 'error' => (string) $decoded['error']];
    }

    return null;
}

/**
 * Fallback: run the Python CLI backend directly so XAMPP works even without manually starting Flask.
 *
 * @return array{status: int, data?: array<string, mixed>, error?: string}
 */
function call_python_cli(string $tmpPath): array
{
    $root = realpath(PROJECT_ROOT) ?: PROJECT_ROOT;
    $pythonExe = find_python_executable();
    $scriptPath = $root . DIRECTORY_SEPARATOR . 'backend' . DIRECTORY_SEPARATOR . 'predict.py';

    if (!is_file($scriptPath)) {
        return [
            'status' => 500,
            'error' => 'Python backend script is missing (backend/predict.py).',
        ];
    }

    $descriptors = [
        0 => ['pipe', 'r'],
        1 => ['pipe', 'w'],
        2 => ['pipe', 'w'],
    ];

    $process = proc_open(
        [$pythonExe, '-B', $scriptPath, $tmpPath],
        $descriptors,
        $pipes,
        $root
    );

    if (!is_resource($process)) {
        return [
            'status' => 500,
            'error' => 'Could not start the Python analysis backend.',
        ];
    }

    fclose($pipes[0]);
    $stdout = stream_get_contents($pipes[1]);
    $stderr = stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);

    $exitCode = proc_close($process);

    $decoded = null;
    if (is_string($stdout)) {
        $trimmed = trim($stdout);
        $decoded = json_decode($trimmed, true);
        if (!is_array($decoded) && $trimmed !== '') {
            $lines = preg_split('/\r\n|\r|\n/', $trimmed) ?: [];
            for ($i = count($lines) - 1; $i >= 0; $i--) {
                $candidate = json_decode(trim($lines[$i]), true);
                if (is_array($candidate)) {
                    $decoded = $candidate;
                    break;
                }
            }
        }
    }

    if ($exitCode === 0 && is_array($decoded) && !empty($decoded['ok'])) {
        return ['status' => 200, 'data' => $decoded];
    }

    if ($exitCode === 2 && is_array($decoded) && isset($decoded['error'])) {
        return ['status' => 400, 'error' => (string) $decoded['error']];
    }

    if (is_string($stderr) && trim($stderr) !== '') {
        error_log('[SkinDiseaseDetection Python Backend] ' . trim($stderr));
    }

    $message = (is_array($decoded) && isset($decoded['error']))
        ? (string) $decoded['error']
        : 'The Python analysis backend could not process this image.';

    return ['status' => 500, 'error' => $message];
}

/**
 * Analyze a skin photo through the Python backend (Flask API if running, else Python CLI).
 *
 * @return array{status: int, data?: array<string, mixed>, error?: string}
 */
function analyze_skin_photo(string $tmpPath, string $originalName = 'upload.jpg'): array
{
    $apiResult = call_python_http_api($tmpPath, $originalName);
    if ($apiResult !== null) {
        return $apiResult;
    }

    return call_python_cli($tmpPath);
}
