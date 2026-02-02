import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "IXIWORKS.ControlNetPreprocessor",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ControlNetPreprocessor") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);

            const self = this;
            const preprocessorWidget = this.widgets.find(
                (w) => w.name === "preprocessor"
            );
            if (!preprocessorWidget) return;

            const origCallback = preprocessorWidget.callback;
            preprocessorWidget.callback = function (value) {
                if (origCallback) origCallback.call(this, value);
                self._updateCannyWidgets(value);
            };

            this._updateCannyWidgets(preprocessorWidget.value);
        };

        nodeType.prototype._updateCannyWidgets = function (preprocessor) {
            const isCanny = preprocessor === "canny";
            for (const w of this.widgets) {
                if (w.name === "low_threshold" || w.name === "high_threshold") {
                    if (isCanny) {
                        w.type = "number";
                        w.computeSize = undefined;
                    } else {
                        w.type = "hidden";
                        w.computeSize = () => [0, -4];
                    }
                }
            }
            this.setSize(this.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const preprocessorWidget = this.widgets && this.widgets.find(
                (w) => w.name === "preprocessor"
            );
            if (preprocessorWidget) {
                this._updateCannyWidgets(preprocessorWidget.value);
            }
        };
    },
});

app.registerExtension({
    name: "IXIWORKS.DiffSynthControlnetAdvanced",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "DiffSynthControlnetAdvanced") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);

            const self = this;
            const fadeWidget = this.widgets.find((w) => w.name === "fade");
            if (!fadeWidget) return;

            const origCallback = fadeWidget.callback;
            fadeWidget.callback = function (value) {
                if (origCallback) origCallback.call(this, value);
                self._updateFadeWidgets(value);
            };

            // Strength curve graph widget (interactive)
            this.widgets.push({
                name: "strength_graph",
                type: "custom",
                value: 0,
                options: { serialize: false },

                _getGraphParams: function (node, widgetWidth, y, h) {
                    const pad = 15;
                    const gx = pad;
                    const gy = y + 4;
                    const gw = (widgetWidth || node.size[0]) - pad * 2;
                    const gh = (h || 65) - 8;
                    const maxS = 2.0;

                    const val = (name, def) => {
                        const w = node.widgets.find((w) => w.name === name);
                        return w != null ? w.value : def;
                    };
                    const strength = val("strength", 1.0);
                    const startP = val("start", 0.0);
                    const endP = val("end", 1.0);
                    const fade = val("fade", "none");
                    const low = val("low", 0.0);

                    let s0, s1;
                    if (fade === "fade out") {
                        s0 = strength; s1 = low;
                    } else if (fade === "fade in") {
                        s0 = low; s1 = strength;
                    } else {
                        s0 = strength; s1 = strength;
                    }

                    const sx = gx + startP * gw;
                    const ex = gx + endP * gw;
                    const toY = (v) => gy + gh - Math.min(v / maxS, 1.0) * gh;
                    const fromY = (py) => {
                        const raw = (gy + gh - py) / gh * maxS;
                        return Math.round(Math.max(0, Math.min(maxS, raw)) * 100) / 100;
                    };
                    const fromX = (px) => {
                        const raw = (px - gx) / gw;
                        return Math.round(Math.max(0, Math.min(1, raw)) * 100) / 100;
                    };

                    return {
                        gx, gy, gw, gh, maxS,
                        strength, startP, endP, fade, low,
                        s0, s1, sx, ex, toY, fromY, fromX, val,
                    };
                },

                draw: function (ctx, node, widgetWidth, y, h) {
                    const p = this._getGraphParams(node, widgetWidth, y, h);
                    node._graphWidgetY = y;
                    node._graphWidgetH = h;

                    ctx.save();

                    // Background
                    ctx.fillStyle = "#1a1a2e";
                    ctx.beginPath();
                    ctx.roundRect(p.gx, p.gy, p.gw, p.gh, 4);
                    ctx.fill();
                    ctx.clip();

                    // Subtle grid lines
                    ctx.strokeStyle = "rgba(255,255,255,0.05)";
                    ctx.lineWidth = 1;
                    for (let i = 1; i < 4; i++) {
                        const ly = p.gy + (p.gh / 4) * i;
                        ctx.beginPath();
                        ctx.moveTo(p.gx, ly);
                        ctx.lineTo(p.gx + p.gw, ly);
                        ctx.stroke();
                    }

                    // Fill under curve
                    ctx.fillStyle = "rgba(179,157,219,0.2)";
                    ctx.beginPath();
                    ctx.moveTo(p.sx, p.gy + p.gh);
                    ctx.lineTo(p.sx, p.toY(p.s0));
                    ctx.lineTo(p.ex, p.toY(p.s1));
                    ctx.lineTo(p.ex, p.gy + p.gh);
                    ctx.closePath();
                    ctx.fill();

                    // Strength line
                    ctx.strokeStyle = "#B39DDB";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(p.sx, p.toY(p.s0));
                    ctx.lineTo(p.ex, p.toY(p.s1));
                    ctx.stroke();

                    // Start/end dashed markers
                    ctx.lineWidth = 1;
                    ctx.setLineDash([3, 3]);
                    if (p.startP > 0.01) {
                        ctx.strokeStyle = node._graphDrag === "start"
                            ? "rgba(255,255,255,0.6)"
                            : "rgba(255,255,255,0.2)";
                        ctx.beginPath();
                        ctx.moveTo(p.sx, p.gy);
                        ctx.lineTo(p.sx, p.gy + p.gh);
                        ctx.stroke();
                    }
                    if (p.endP < 0.99) {
                        ctx.strokeStyle = node._graphDrag === "end"
                            ? "rgba(255,255,255,0.6)"
                            : "rgba(255,255,255,0.2)";
                        ctx.beginPath();
                        ctx.moveTo(p.ex, p.gy);
                        ctx.lineTo(p.ex, p.gy + p.gh);
                        ctx.stroke();
                    }
                    ctx.setLineDash([]);

                    // Endpoint dots (larger when dragging)
                    ctx.fillStyle = "#B39DDB";
                    const r0 = node._graphDrag === "s0" ? 5 : 3;
                    ctx.beginPath();
                    ctx.arc(p.sx, p.toY(p.s0), r0, 0, Math.PI * 2);
                    ctx.fill();
                    const r1 = node._graphDrag === "s1" ? 5 : 3;
                    ctx.beginPath();
                    ctx.arc(p.ex, p.toY(p.s1), r1, 0, Math.PI * 2);
                    ctx.fill();

                    // Strength value labels
                    ctx.font = "10px monospace";
                    ctx.fillStyle = "rgba(179,157,219,0.9)";
                    const s0y = p.toY(p.s0);
                    const s1y = p.toY(p.s1);
                    ctx.textAlign = "left";
                    ctx.fillText(
                        p.s0.toFixed(2), p.sx + 6,
                        s0y < p.gy + 14 ? s0y + 14 : s0y - 5
                    );
                    if (Math.abs(p.s0 - p.s1) > 0.01) {
                        ctx.textAlign = "right";
                        ctx.fillText(
                            p.s1.toFixed(2), p.ex - 6,
                            s1y < p.gy + 14 ? s1y + 14 : s1y - 5
                        );
                    }

                    ctx.restore();
                },

                mouse: function (event, pos, node) {
                    if (node._graphWidgetY == null) return false;

                    const p = this._getGraphParams(
                        node, node.size[0],
                        node._graphWidgetY, node._graphWidgetH
                    );
                    const [mx, my] = pos;
                    const hitR = 10;

                    const setVal = (name, v) => {
                        const w = node.widgets.find((w) => w.name === name);
                        if (w) { w.value = v; if (w.callback) w.callback(v); }
                    };

                    const etype = event.type;

                    if (etype === "pointerdown" || etype === "mousedown") {
                        // Hit test: dots first, then markers
                        if (Math.hypot(mx - p.sx, my - p.toY(p.s0)) < hitR) {
                            node._graphDrag = "s0"; return true;
                        }
                        if (Math.hypot(mx - p.ex, my - p.toY(p.s1)) < hitR) {
                            node._graphDrag = "s1"; return true;
                        }
                        if (Math.abs(mx - p.sx) < hitR && my >= p.gy && my <= p.gy + p.gh) {
                            node._graphDrag = "start"; return true;
                        }
                        if (Math.abs(mx - p.ex) < hitR && my >= p.gy && my <= p.gy + p.gh) {
                            node._graphDrag = "end"; return true;
                        }
                        return false;
                    }

                    if ((etype === "pointermove" || etype === "mousemove") && node._graphDrag) {
                        const drag = node._graphDrag;
                        if (drag === "s0") {
                            const v = p.fromY(my);
                            if (p.fade === "fade in") setVal("low", v);
                            else setVal("strength", v);
                        } else if (drag === "s1") {
                            const v = p.fromY(my);
                            if (p.fade === "fade out") setVal("low", v);
                            else setVal("strength", v);
                        } else if (drag === "start") {
                            setVal("start", Math.min(p.fromX(mx), p.val("end", 1)));
                        } else if (drag === "end") {
                            setVal("end", Math.max(p.fromX(mx), p.val("start", 0)));
                        }
                        app.graph.setDirtyCanvas(true, true);
                        return true;
                    }

                    if (etype === "pointerup" || etype === "mouseup") {
                        if (node._graphDrag) {
                            node._graphDrag = null;
                            return true;
                        }
                    }

                    return false;
                },

                computeSize: function () {
                    return [0, 65];
                },
            });

            this._updateFadeWidgets(fadeWidget.value);
        };

        nodeType.prototype._updateFadeWidgets = function (fadeValue) {
            const fadeOn = fadeValue !== "none";
            for (const w of this.widgets) {
                if (w.name === "strength") {
                    // fade off → show strength widget, fade on → hide (graph controls it)
                    if (fadeOn) {
                        w.type = "hidden";
                        w.computeSize = () => [0, -4];
                    } else {
                        w.type = "number";
                        w.computeSize = undefined;
                    }
                } else if (w.name === "low") {
                    // fade on → show low widget, fade off → hide
                    if (fadeOn) {
                        w.type = "number";
                        w.computeSize = undefined;
                    } else {
                        w.type = "hidden";
                        w.computeSize = () => [0, -4];
                    }
                } else if (w.name === "strength_graph") {
                    // fade on → show graph, fade off → hide
                    if (fadeOn) {
                        w.type = "custom";
                        w.computeSize = w._origComputeSize || (() => [0, 65]);
                    } else {
                        if (!w._origComputeSize) w._origComputeSize = w.computeSize;
                        w.type = "hidden";
                        w.computeSize = () => [0, -4];
                    }
                }
            }
            this.setSize(this.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const fadeWidget = this.widgets && this.widgets.find(
                (w) => w.name === "fade"
            );
            if (fadeWidget) {
                this._updateFadeWidgets(fadeWidget.value);
            }
        };
    },
});
