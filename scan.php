<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/layout.php';
require_once __DIR__ . '/includes/python_client.php';

$uploadError = '';
$analysis = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!isset($_FILES['file'])) {
        $contentLength = (int) ($_SERVER['CONTENT_LENGTH'] ?? 0);
        http_response_code(400);
        $uploadError = $contentLength > 0
            ? 'The uploaded image is too large.'
            : 'Choose an image to upload.';
    } else {
        $file = $_FILES['file'];
        $errorCode = (int) ($file['error'] ?? UPLOAD_ERR_NO_FILE);
        $originalName = (string) ($file['name'] ?? '');
        $tmpPath = (string) ($file['tmp_name'] ?? '');

        if ($errorCode === UPLOAD_ERR_NO_FILE || $originalName === '') {
            http_response_code(400);
            $uploadError = 'Choose an image to upload.';
        } elseif ($errorCode === UPLOAD_ERR_INI_SIZE || $errorCode === UPLOAD_ERR_FORM_SIZE) {
            http_response_code(400);
            $uploadError = 'The uploaded image is too large.';
        } elseif ($errorCode !== UPLOAD_ERR_OK || $tmpPath === '' || !is_uploaded_file($tmpPath)) {
            http_response_code(400);
            $uploadError = 'The uploaded image could not be received.';
        } elseif (filesize($tmpPath) === 0) {
            http_response_code(400);
            $uploadError = 'The uploaded image is empty.';
        } else {
            if (function_exists('set_time_limit')) {
                @set_time_limit(120);
            }
            $response = analyze_skin_photo($tmpPath, $originalName);
            if ($response['status'] === 200 && isset($response['data'])) {
                $analysis = $response['data'];
                header('Cache-Control: no-store');
                header('Referrer-Policy: no-referrer');
            } else {
                http_response_code($response['status']);
                $uploadError = $response['error'] ?? 'Upload a readable image.';
            }
        }
    }
}

if ($analysis !== null):
    $prediction = (string) $analysis['prediction'];
    $hasMatch = $prediction !== 'Uncertain';
    $score = (float) $analysis['score'];
    $scoreText = number_format($score, 1, '.', '');
    $band = (string) $analysis['band'];
    $region = is_array($analysis['region'] ?? null) ? $analysis['region'] : null;
    $specificCondition = is_array($analysis['specific_condition'] ?? null) ? $analysis['specific_condition'] : null;
    $topCategories = is_array($analysis['top_categories'] ?? null) ? $analysis['top_categories'] : [];
    $topSpecific = is_array($analysis['top_specific'] ?? null) ? $analysis['top_specific'] : [];
    $appearance = isset($analysis['appearance']) && is_string($analysis['appearance']) ? $analysis['appearance'] : null;
    $reference = isset($analysis['reference']) && is_string($analysis['reference']) ? $analysis['reference'] : null;
    $explanation = (string) $analysis['explanation'];
    $record = $analysis['record'];
    $photoRatio = (float) $analysis['photo_ratio'];
    $imagePreview = (string) $analysis['image_preview'];
    $recordJson = json_encode($record, JSON_HEX_TAG | JSON_HEX_AMP | JSON_UNESCAPED_SLASHES);

    render_page_start([
        'title' => 'Analysis Result | Skin Disease Detection',
        'page_class' => 'analysis-theme',
        'show_header' => false,
        'scripts' => ['static/js/results.js'],
    ]);
    ?>
<main class="analysis-page shell">
    <header class="analysis-heading">
        <?= svg_icon('sparkles') ?>
        <div>
            <h1>Analysis Result</h1>
            <p>AI-powered preliminary analysis of your uploaded skin photo.</p>
        </div>
    </header>

    <div class="analysis-layout">
        <section class="analysis-panel analysis-image" aria-labelledby="result-preview-title">
            <h2 id="result-preview-title" class="analysis-section-title"><?= svg_icon('photo') ?> Analyzed Image</h2>
            <figure class="result-figure">
                <div class="result-photo" style="--photo-ratio: <?= e((string) $photoRatio) ?>">
                    <img src="data:image/jpeg;base64,<?= e($imagePreview) ?>" alt="Uploaded skin photo">
                </div>
            </figure>
            <a class="secondary-button analysis-another" href="scan.php"><?= svg_icon('refresh') ?> Analyze Another</a>
        </section>

        <section class="analysis-panel analysis-report" aria-labelledby="result-title">
            <header class="prediction-banner">
                <span class="prediction-icon"><?= svg_icon('label') ?></span>
                <div>
                    <p>Preliminary Analysis</p>
                    <h2 class="prediction" id="result-title"><?= $hasMatch ? 'Possible: ' . e($prediction) : 'No reliable match' ?></h2>
                    <?php if ($specificCondition !== null && !empty($specificCondition['name'])): ?>
                    <p class="specific-pattern">Closest specific pattern: <strong><?= e((string) $specificCondition['name']) ?></strong></p>
                    <?php endif; ?>
                </div>
            </header>

            <section class="confidence-strip" aria-labelledby="confidence-title">
                <div class="confidence-label">
                    <h3 id="confidence-title"><?= $hasMatch ? 'Model confidence' : 'Highest model score' ?></h3>
                    <button id="confidence-info" class="confidence-info" type="button" aria-label="About model confidence"
                            aria-haspopup="dialog" aria-controls="confidence-dialog"><?= svg_icon('info') ?></button>
                </div>
                <div class="confidence-gauge">
                    <meter class="confidence-meter" min="0" max="100" value="<?= e($scoreText) ?>" aria-labelledby="confidence-title"
                           aria-valuetext="<?= e($scoreText) ?> percent out of 100, <?= e(strtolower($band)) ?> range"></meter>
                    <div class="confidence-scale" aria-hidden="true"><span>0%</span><span>100% max</span></div>
                </div>
                <div class="confidence-chip"><strong><?= e($scoreText) ?>%</strong><span class="confidence-band"><?= e($band) ?></span></div>
            </section>

            <?php if (count($topCategories) >= 2): ?>
            <section class="differential-strip" aria-labelledby="differential-title">
                <h3 id="differential-title" class="differential-heading"><?= $hasMatch ? 'Top differential possibilities' : 'Closest possibilities (not a reliable match)' ?></h3>
                <ol class="differential-list">
                    <?php foreach (array_slice($topCategories, 0, 3) as $item):
                        $itemName = (string) ($item['name'] ?? '');
                        $itemScore = (float) ($item['score'] ?? 0.0);
                        $itemPct = number_format($itemScore, 1, '.', '');
                    ?>
                    <li class="differential-item">
                        <span class="differential-name"><?= e($itemName) ?></span>
                        <span class="differential-bar" aria-hidden="true"><span style="width: <?= e((string) min(100.0, max(0.0, $itemScore))) ?>%"></span></span>
                        <strong class="differential-score"><?= e($itemPct) ?>%</strong>
                    </li>
                    <?php endforeach; ?>
                </ol>
                <?php if (!$hasMatch): ?>
                <p class="specific-hints">None of these scored high enough to count as a match. They are listed for reference only.</p>
                <?php endif; ?>
                <?php if (count($topSpecific) > 0): ?>
                <p class="specific-hints"><strong>Specific visual matches:</strong>
                    <?php
                    $hintParts = [];
                    foreach (array_slice($topSpecific, 0, 3) as $spec) {
                        $hintParts[] = e((string) ($spec['name'] ?? '')) . ' (' . e(number_format((float) ($spec['score'] ?? 0.0), 1, '.', '')) . '%)';
                    }
                    echo implode(' · ', $hintParts);
                    ?>
                </p>
                <?php endif; ?>
            </section>
            <?php endif; ?>

            <div class="analysis-details">
                <section aria-labelledby="condition-title">
                    <h3 id="condition-title" class="analysis-section-title"><?= svg_icon('document') ?> <?= $hasMatch ? 'About this condition' : 'About this result' ?></h3>
                    <?php if ($appearance !== null && $reference !== null): ?>
                    <p><?= e($appearance) ?> <a href="<?= e($reference) ?>" rel="noreferrer">Condition reference</a></p>
                    <?php elseif ($hasMatch): ?>
                    <p>The model selected this category from the conditions it was trained to recognize.</p>
                    <?php endif; ?>
                    <p class="analysis-evidence"><strong><?= $hasMatch ? 'Why this label' : 'Why no match' ?></strong> <?= e($explanation) ?></p>
                </section>

                <section aria-labelledby="care-title">
                    <h3 id="care-title" class="analysis-section-title"><?= svg_icon('shield') ?> General Care Tips</h3>
                    <ul class="care-list">
                        <li><?= svg_icon('check') ?><a href="https://www.aad.org/public/everyday-care/skin-care-secrets/routine/healthier-looking-skin" rel="noreferrer">Cleanse gently and avoid scrubbing.</a></li>
                        <li><?= svg_icon('check') ?><a href="https://www.aad.org/public/diseases/skin-cancer/how-to-prevent" rel="noreferrer">Protect exposed skin from the sun.</a></li>
                        <li><?= svg_icon('check') ?><a href="https://www.aad.org/public/diseases/skin-cancer/check-skin" rel="noreferrer">Have a new, changing, or bleeding spot checked.</a></li>
                    </ul>
                </section>
            </div>

            <aside class="medical-notice" aria-labelledby="disclaimer-title">
                <?= svg_icon('shield') ?>
                <h3 id="disclaimer-title">Important disclaimer</h3>
                <p><strong>A diagnosis requires a clinical exam.</strong> Ask a qualified healthcare professional for evaluation and treatment advice.</p>
            </aside>

            <footer class="analysis-actions">
                <a class="secondary-button" id="download-result" download="skin-analysis.txt" href="data:text/plain;charset=utf-8;base64,<?= e((string) $record['report']) ?>"><?= svg_icon('download') ?> Download Result</a>
                <a class="primary-button" id="view-results" href="results.php"><?= svg_icon('history') ?> View Results</a>
            </footer>
            <p id="history-save-status" class="history-status" role="status" hidden></p>
        </section>
    </div>

    <dialog id="confidence-dialog" class="camera-dialog confidence-dialog" aria-labelledby="confidence-dialog-title">
        <h2 id="confidence-dialog-title">About model confidence</h2>
        <p>This score shows how strongly the model favored its leading category. The scale runs from 0% to 100%.</p>
        <dl>
            <div><dt>Low</dt><dd>Below 50%</dd></div>
            <div><dt>Moderate</dt><dd>50% to 79.9%</dd></div>
            <div><dt>High</dt><dd>80% to 100%</dd></div>
        </dl>
        <p>A diagnosis requires clinical assessment at any score.</p>
        <form method="dialog"><button class="secondary-button" autofocus>Close</button></form>
    </dialog>
    <script type="application/json" id="analysis-record"><?= $recordJson !== false ? $recordJson : '{}' ?></script>
</main>
<?php
    render_page_end();
else:
    render_page_start([
        'title' => 'Skin Disease Detection',
        'scripts' => ['static/js/scanner.js'],
        'header_actions' => '<a class="results-link" href="results.php">' . svg_icon('history') . ' View Results</a>',
    ]);
    ?>
<main class="shell">
    <form class="workspace" action="scan.php" method="post" enctype="multipart/form-data">
        <section class="source-panel" aria-labelledby="scan-title">
            <h1 id="scan-title">Add a skin photo</h1>
            <input id="skin-photo" type="file" name="file" accept="image/*,.jpg,.jpeg,.png,.webp,.gif,.bmp,.tif,.tiff,.avif,.ico,.jfif,.heic,.heif" hidden>
            <div class="source-buttons">
                <button class="source-button" id="choose-photo" type="button">
                    <span class="source-icon"><?= svg_icon('upload') ?></span>
                    <span class="source-label"><strong>Upload Image</strong><span>Choose any image file</span></span>
                    <span class="source-next"><?= svg_icon('next') ?></span>
                </button>
                <button class="source-button" id="open-camera" type="button">
                    <span class="source-icon"><?= svg_icon('camera') ?></span>
                    <span class="source-label"><strong>Open Camera</strong><span>Take a photo with your device</span></span>
                    <span class="source-next"><?= svg_icon('next') ?></span>
                </button>
            </div>
            <p class="lighting-tip"><?= svg_icon('sun') ?><span>Frame the skin area close up and use even lighting.</span></p>
            <p class="notice" id="scanner-message" role="status"<?= $uploadError === '' ? ' hidden' : '' ?>><?= e($uploadError) ?></p>
        </section>

        <section class="preview-panel" aria-labelledby="preview-title">
            <div class="preview-heading">
                <h2 id="preview-title">Photo preview</h2>
                <button class="icon-button" id="remove-photo" type="button" aria-label="Remove photo" title="Remove photo"><?= svg_icon('trash') ?></button>
            </div>
            <figure class="photo-frame">
                <img id="photo-preview" class="example-photo" src="static/images/example-skin-photo.png"
                     alt="Example forearm photo" width="1536" height="1024">
                <figcaption class="example-label" id="example-label">Example photo</figcaption>
                <div class="empty-photo" id="empty-photo" hidden>
                    <?= svg_icon('photo') ?><p>Choose or take a photo to preview it.</p>
                </div>
            </figure>
            <button class="primary-button" id="check-image" type="submit" disabled>
                Check Image <?= svg_icon('arrow') ?>
            </button>
        </section>
    </form>

    <dialog class="camera-dialog" id="camera-dialog" aria-labelledby="camera-title">
        <div class="preview-heading">
            <h2 id="camera-title">Take a photo</h2>
            <form method="dialog">
                <button class="icon-button" aria-label="Close camera" title="Close camera"><?= svg_icon('close') ?></button>
            </form>
        </div>
        <video id="camera-video" autoplay muted playsinline hidden></video>
        <p class="notice" id="camera-message" role="status"></p>
        <button class="primary-button" id="capture-photo" type="button" disabled>Take Photo <?= svg_icon('camera') ?></button>
    </dialog>
</main>
<?php
    render_page_end();
endif;
