import { app } from "../../scripts/app.js";

const WILDCARD_NODES = ["SwitchBoolean", "SwitchCase", "Bypass"];

function getSlotColor(type) {
    const canvas = app.canvas;
    const sources = [
        canvas?.default_connection_color_byType,
        typeof LGraphCanvas !== "undefined" ? LGraphCanvas.link_type_colors : null,
    ];
    for (const src of sources) {
        if (src?.[type]) return src[type];
    }
    return null;
}

function detectType(node) {
    if (!node.graph || !node.inputs) return null;
    for (const input of node.inputs) {
        if (input.link == null) continue;
        const link = node.graph.links[input.link];
        if (!link) continue;
        const src = node.graph.getNodeById(link.origin_id);
        if (!src?.outputs) continue;
        const srcSlot = src.outputs[link.origin_slot];
        if (srcSlot?.type && srcSlot.type !== "*") return srcSlot.type;
        // Chain: source is also a wildcard node with a detected type
        if (src._ixiSlotType) return src._ixiSlotType;
    }
    return null;
}

function applyColor(node) {
    const type = detectType(node);
    node._ixiSlotType = type;
    const color = type ? getSlotColor(type) : null;

    if (node.outputs?.[0]) {
        if (color) {
            node.outputs[0].color_on = color;
            node.outputs[0].color_off = color;
        } else {
            delete node.outputs[0].color_on;
            delete node.outputs[0].color_off;
        }
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "IXIWORKS.WildcardSlotColor",

    async nodeCreated(node) {
        if (!WILDCARD_NODES.includes(node.comfyClass)) return;

        const origConn = node.onConnectionsChange;
        node.onConnectionsChange = function (slotType, slotIndex, isConnect, linkInfo, ioSlot) {
            if (origConn) origConn.apply(this, arguments);
            if (slotType === 1) applyColor(this);
        };

        const origCfg = node.onConfigure;
        node.onConfigure = function (info) {
            if (origCfg) origCfg.apply(this, arguments);
            requestAnimationFrame(() => applyColor(this));
        };
    },
});
