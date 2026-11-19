"use strict";

const historyKey = "skin-analysis-history";

function isSummary(record) {
    return record && typeof record.id === "string" && record.id &&
        typeof record.prediction === "string" && Number.isFinite(record.score) &&
        record.score >= 0 && record.score <= 100 &&
        ["Low", "Moderate", "High"].includes(record.band) &&
        typeof record.explanation === "string" &&
        (record.appearance == null || typeof record.appearance === "string") &&
        typeof record.report === "string" && /^[A-Za-z0-9+/]+={0,2}$/.test(record.report);
}

function readHistory() {
    const records = JSON.parse(sessionStorage.getItem(historyKey) ?? "[]");
    if (!Array.isArray(records) || !records.every(isSummary)) {
        throw new Error("Invalid saved results");
    }
    return records;
}

const result = document.getElementById("analysis-record");
if (result) {
    try {
        const record = JSON.parse(result.textContent);
        if (!isSummary(record)) throw new Error("Invalid analysis");
        const records = readHistory();
        if (!records.some(saved => saved.id === record.id)) {
            const { id, prediction, score, band, explanation, appearance, report, created_at } = record;
            records.unshift({ id, prediction, score, band, explanation, appearance, report, created_at });
            sessionStorage.setItem(historyKey, JSON.stringify(records.slice(0, 20)));
        }
    } catch {
        const status = document.getElementById("history-save-status");
        if (status) {
            status.textContent = "Results history is unavailable in this tab.";
            status.hidden = false;
        }
    }
}

const confidenceInfo = document.getElementById("confidence-info");
const confidenceDialog = document.getElementById("confidence-dialog");
if (confidenceInfo && confidenceDialog) {
    confidenceInfo.addEventListener("click", () => confidenceDialog.showModal());
}

const list = document.getElementById("history-list");
if (list) {
    const status = document.getElementById("history-status");
    const emptyMessage = "No results saved in this tab yet.";
    const template = document.getElementById("history-item-template");

    function renderHistory() {
        list.replaceChildren();
        try {
            const records = readHistory();
            status.textContent = records.length ? "" : emptyMessage;
            for (const record of records) {
                const row = template.content.firstElementChild.cloneNode(true);
                const hasMatch = record.prediction !== "Uncertain";
                row.querySelector("h2").textContent = !hasMatch
                    ? "No reliable match"
                    : `Possible: ${record.prediction}`;
                const score = `${record.score.toFixed(1)}% / 100%`;
                row.querySelector(".history-confidence").textContent = hasMatch
                    ? `${score} · ${record.band} confidence`
                    : `Highest model score ${score}`;
                const date = new Date(record.created_at ?? "");
                if (Number.isFinite(date.getTime())) {
                    const time = row.querySelector("time");
                    time.dateTime = date.toISOString();
                    time.textContent = date.toLocaleString();
                    time.hidden = false;
                }
                row.querySelector(".history-explanation").textContent = record.explanation;
                if (record.appearance) {
                    const appearance = row.querySelector(".history-appearance");
                    appearance.textContent = record.appearance;
                    appearance.hidden = false;
                }
                row.querySelector("a[download]").href = `data:text/plain;charset=utf-8;base64,${record.report}`;
                const remove = row.querySelector("button");
                remove.setAttribute("aria-label", `Remove result for ${record.prediction}`);
                remove.addEventListener("click", () => {
                    try {
                        const remaining = readHistory().filter(saved => saved.id !== record.id);
                        sessionStorage.setItem(historyKey, JSON.stringify(remaining));
                    } catch {
                        status.textContent = "This result could not be removed.";
                        return;
                    }
                    const next = row.nextElementSibling ?? row.previousElementSibling;
                    row.remove();
                    status.textContent = list.children.length ? "Result removed." : emptyMessage;
                    (next?.querySelector("button") ?? status).focus();
                });
                list.appendChild(row);
            }
        } catch {
            status.textContent = "Results history is unavailable in this tab.";
        }
    }

    renderHistory();
    window.addEventListener("pageshow", event => {
        if (event.persisted) renderHistory();
    });
}
