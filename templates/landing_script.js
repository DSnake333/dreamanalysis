// Get the sections and the dots
const sections = document.querySelectorAll('.section');
const dots = document.querySelectorAll('.dot');

// Define a variable to store the current section index
let current = 0;

// Define a function to switch the sections
function switchSection(newIndex) {
    // Remove the active class from the current section and dot
    sections[current].classList.remove('active');
    dots[current].classList.remove('active');

    // Add the active class to the new section and dot
    sections[newIndex].classList.add('active');
    dots[newIndex].classList.add('active');

    // Update the current index
    current = newIndex;
}

// Add an event listener for the mouse wheel
window.addEventListener('wheel', (e) => {
    // Get the direction of the wheel
    const direction = e.deltaY > 0 ? 1 : -1;

    // If the direction is positive, increment the current index
    if (direction > 0) {
        current = current < sections.length - 1 ? current + 1 : 0;
    }

    // If the direction is negative, decrement the current index
    if (direction < 0) {
        current = current > 0 ? current - 1 : sections.length - 1;
    }

    // Switch the section based on the current index
    switchSection(current);
});

// Add an event listener for the click on the dots
dots.forEach((dot, index) => {
    dot.addEventListener('click', () => {
        // Switch the section based on the index of the clicked dot
        switchSection(index);
    });
});
