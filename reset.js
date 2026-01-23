module.exports = {
    run: [
        // Remove virtual environment
        {
            method: "fs.rm",
            params: {
                path: "app/env"
            }
        },
        // Remove cache
        {
            method: "fs.rm",
            params: {
                path: "app/cache"
            }
        },
        {
            method: "input",
            params: {
                title: "Reset Complete!",
                description: "Virtual environment and cache have been removed. Run 'Install' to set up again.\n\nNote: Your audiobook library was preserved."
            }
        }
    ]
}
