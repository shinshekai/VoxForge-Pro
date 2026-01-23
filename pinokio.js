const path = require('path')
module.exports = {
    version: "5.0",
    title: "VoxForge Pro",
    description: "Premium AI-Powered Audiobook Generator",
    icon: "icon.png",
    menu: async (kernel, info) => {
        let installed = info.exists("app/env")
        let running = {
            install: info.running("install.js"),
            start: info.running("start.js"),
            update: info.running("update.js"),
            reset: info.running("reset.js"),
            link: info.running("link.js")
        }

        // Installation in progress
        if (running.install) {
            return [{
                default: true,
                icon: "fa-solid fa-plug",
                text: "Installing",
                href: "install.js"
            }]
        }

        // App is installed
        if (installed) {
            // App is running
            if (running.start) {
                let local = info.local("start.js")
                if (local && local.url) {
                    return [{
                        default: true,
                        icon: "fa-solid fa-rocket",
                        text: "Open VoxForge Pro",
                        href: local.url
                    }, {
                        icon: "fa-solid fa-terminal",
                        text: "Terminal",
                        href: "start.js"
                    }]
                } else {
                    return [{
                        default: true,
                        icon: "fa-solid fa-terminal",
                        text: "Terminal",
                        href: "start.js"
                    }]
                }
            }

            // Update in progress
            if (running.update) {
                return [{
                    default: true,
                    icon: "fa-solid fa-rotate",
                    text: "Updating",
                    href: "update.js"
                }]
            }

            // Reset in progress
            if (running.reset) {
                return [{
                    default: true,
                    icon: "fa-solid fa-trash",
                    text: "Resetting",
                    href: "reset.js"
                }]
            }

            // Deduplication in progress
            if (running.link) {
                return [{
                    default: true,
                    icon: "fa-solid fa-link",
                    text: "Deduplicating",
                    href: "link.js"
                }]
            }

            // App is ready (not running)
            return [{
                default: true,
                icon: "fa-solid fa-play",
                text: "Start",
                href: "start.js"
            }, {
                icon: "fa-solid fa-rotate",
                text: "Update",
                href: "update.js"
            }, {
                icon: "fa-solid fa-plug",
                text: "Reinstall",
                href: "install.js"
            }, {
                icon: "fa-solid fa-file-zipper",
                text: "<div><strong>Save Disk Space</strong><div>Deduplicates redundant library files</div></div>",
                href: "link.js"
            }, {
                icon: "fa-regular fa-circle-xmark",
                text: "<div><strong>Reset</strong><div>Revert to pre-install state</div></div>",
                href: "reset.js",
                confirm: "Are you sure you wish to reset? This will remove the virtual environment but preserve your audiobook library."
            }]
        }

        // Not installed - show Install button
        return [{
            default: true,
            icon: "fa-solid fa-plug",
            text: "Install",
            href: "install.js"
        }]
    }
}
