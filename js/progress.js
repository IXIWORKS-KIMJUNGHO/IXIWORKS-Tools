import { app } from "../../scripts/app.js";

// Progress bar state
let progressState = {
    jobId: null,
    redisUrl: "redis://localhost:6379",
    polling: false,
    interval: null,
};

// Create progress bar UI
function createProgressBar() {
    // Check if already exists
    if (document.getElementById("ixiworks-progress-container")) {
        return;
    }

    const container = document.createElement("div");
    container.id = "ixiworks-progress-container";
    container.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 32px;
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-bottom: 1px solid #0f3460;
        display: none;
        align-items: center;
        padding: 0 16px;
        z-index: 10000;
        font-family: 'Segoe UI', system-ui, sans-serif;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    `;

    container.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px; width: 100%;">
            <span style="color: #B39DDB; font-size: 12px; font-weight: 500;">IXIWORKS</span>

            <div id="ixiworks-progress-bar-bg" style="
                flex: 1;
                height: 8px;
                background: rgba(255,255,255,0.1);
                border-radius: 4px;
                overflow: hidden;
            ">
                <div id="ixiworks-progress-bar-fill" style="
                    width: 0%;
                    height: 100%;
                    background: linear-gradient(90deg, #B39DDB 0%, #9575CD 100%);
                    border-radius: 4px;
                    transition: width 0.3s ease;
                "></div>
            </div>

            <span id="ixiworks-progress-percent" style="
                color: #fff;
                font-size: 12px;
                min-width: 50px;
                text-align: right;
            ">0%</span>

            <span id="ixiworks-progress-batch" style="
                color: rgba(255,255,255,0.7);
                font-size: 11px;
                min-width: 60px;
            ">0/0</span>

            <span id="ixiworks-progress-step" style="
                color: rgba(255,255,255,0.5);
                font-size: 11px;
                min-width: 60px;
            ">0/0</span>

            <button id="ixiworks-progress-close" style="
                background: none;
                border: none;
                color: rgba(255,255,255,0.5);
                cursor: pointer;
                font-size: 16px;
                padding: 4px 8px;
                border-radius: 4px;
            ">✕</button>
        </div>
    `;

    document.body.appendChild(container);

    // Close button
    document.getElementById("ixiworks-progress-close").addEventListener("click", () => {
        hideProgressBar();
        stopPolling();
    });

    // Adjust ComfyUI layout
    const style = document.createElement("style");
    style.id = "ixiworks-progress-style";
    style.textContent = `
        body.ixiworks-progress-active {
            padding-top: 32px !important;
        }
        body.ixiworks-progress-active .comfy-menu {
            top: 32px !important;
        }
    `;
    document.head.appendChild(style);
}

function showProgressBar() {
    const container = document.getElementById("ixiworks-progress-container");
    if (container) {
        container.style.display = "flex";
        document.body.classList.add("ixiworks-progress-active");
    }
}

function hideProgressBar() {
    const container = document.getElementById("ixiworks-progress-container");
    if (container) {
        container.style.display = "none";
        document.body.classList.remove("ixiworks-progress-active");
    }
}

function updateProgressBar(data) {
    const fill = document.getElementById("ixiworks-progress-bar-fill");
    const percent = document.getElementById("ixiworks-progress-percent");
    const batch = document.getElementById("ixiworks-progress-batch");
    const step = document.getElementById("ixiworks-progress-step");

    if (fill && percent && batch && step) {
        const progress = (data.progress || 0) * 100;
        fill.style.width = `${progress}%`;
        percent.textContent = `${progress.toFixed(1)}%`;
        batch.textContent = `Batch ${data.batch_current || 0}/${data.batch_total || 0}`;
        step.textContent = `Step ${data.step_current || 0}/${data.step_total || 0}`;

        // Change color when complete
        if (data.status === "completed" || data.progress >= 1) {
            fill.style.background = "linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%)";
        } else {
            fill.style.background = "linear-gradient(90deg, #B39DDB 0%, #9575CD 100%)";
        }
    }
}

async function fetchProgress() {
    if (!progressState.jobId) return;

    try {
        const url = `/ixiworks/progress/${progressState.jobId}?redis_url=${encodeURIComponent(progressState.redisUrl)}`;
        const response = await fetch(url);

        if (response.ok) {
            const data = await response.json();
            updateProgressBar(data);

            // Auto-hide on completion after delay
            if (data.status === "completed" || data.progress >= 1) {
                setTimeout(() => {
                    stopPolling();
                }, 3000);
            }
        }
    } catch (e) {
        console.warn("[IXIWORKS] Progress fetch failed:", e);
    }
}

function startPolling(jobId, redisUrl) {
    progressState.jobId = jobId;
    progressState.redisUrl = redisUrl || progressState.redisUrl;
    progressState.polling = true;

    showProgressBar();
    updateProgressBar({ progress: 0, batch_current: 0, batch_total: 0, step_current: 0, step_total: 0 });

    // Poll every 500ms
    if (progressState.interval) {
        clearInterval(progressState.interval);
    }
    progressState.interval = setInterval(fetchProgress, 500);
    fetchProgress(); // Initial fetch
}

function stopPolling() {
    progressState.polling = false;
    if (progressState.interval) {
        clearInterval(progressState.interval);
        progressState.interval = null;
    }
}

// Expose global functions for manual control
window.ixiworksProgress = {
    start: startPolling,
    stop: stopPolling,
    show: showProgressBar,
    hide: hideProgressBar,
};

// Register extension
app.registerExtension({
    name: "IXIWORKS.ProgressBar",

    async setup() {
        createProgressBar();

        // Listen for ModelProgressTracker node execution
        const originalQueuePrompt = app.queuePrompt;
        app.queuePrompt = async function (...args) {
            // Check if workflow has ModelProgressTracker
            const nodes = app.graph._nodes;
            for (const node of nodes) {
                if (node.type === "ModelProgressTracker") {
                    const jobIdWidget = node.widgets?.find(w => w.name === "job_id");
                    const redisUrlWidget = node.widgets?.find(w => w.name === "redis_url");

                    if (jobIdWidget?.value) {
                        startPolling(jobIdWidget.value, redisUrlWidget?.value);
                    }
                    break;
                }
            }

            return originalQueuePrompt.apply(this, args);
        };
    },
});
