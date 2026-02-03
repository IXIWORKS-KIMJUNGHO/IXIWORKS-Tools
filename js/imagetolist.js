import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "IXIWORKS.ImageToList",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ImageToList") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);

            const self = this;
            const countWidget = this.widgets.find((w) => w.name === "count");
            if (!countWidget) return;

            const origCallback = countWidget.callback;
            countWidget.callback = function (value) {
                if (origCallback) origCallback.call(this, value);
                self._updateInputSlots(value);
            };

            // Initial setup
            this._updateInputSlots(countWidget.value);
        };

        nodeType.prototype._updateInputSlots = function (count) {
            // Remove extra inputs beyond count
            while (this.inputs && this.inputs.length > count) {
                this.removeInput(this.inputs.length - 1);
            }
            // Add missing inputs up to count
            const currentCount = this.inputs ? this.inputs.length : 0;
            for (let i = currentCount + 1; i <= count; i++) {
                this.addInput(`image_${i}`, "IMAGE");
            }
            this.setSize(this.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const countWidget = this.widgets && this.widgets.find((w) => w.name === "count");
            if (countWidget) {
                // Delay to ensure node is fully configured
                setTimeout(() => {
                    this._updateInputSlots(countWidget.value);
                }, 0);
            }
        };
    }
});
