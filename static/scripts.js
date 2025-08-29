document.addEventListener("DOMContentLoaded", function() {
    let sidebar = document.querySelector("[data-testid='stSidebar']");
    let toggleButton = document.querySelector(".toggle-button");

    // Función para abrir/cerrar la barra lateral
    if (toggleButton) {
        toggleButton.addEventListener("click", function() {
            const isHidden = sidebar.style.transform === "translateX(-260px)";
            sidebar.style.transform = isHidden ? "translateX(0)" : "translateX(-260px)";
        });
    }
});

