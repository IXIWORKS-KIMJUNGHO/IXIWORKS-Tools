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
