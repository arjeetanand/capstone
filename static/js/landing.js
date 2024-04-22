// Define bot responses
// const botResponses = {
//     "hi": "Hello there!",
//     "how are you": "I'm just a bot, but thanks for asking!",
//     "goodbye": "Goodbye! Have a great day!",
//     // Add more responses as needed
// };

// // Function to send user message and receive bot response
// function sendMessage() {
//     const userInput = document.querySelector('.user-input');
//     const userMessage = userInput.value.trim().toLowerCase();

//     if (userMessage !== '') {
//         displayMessage('user', userMessage);
//         userInput.value = ''; // Clear input field

//         // Check if there's a predefined response for the user message
//         const botResponse = botResponses[userMessage];
//         if (botResponse) {
//             displayMessage('bot', botResponse);
//         } else {
//             displayMessage('bot', "I'm sorry, I don't understand that.");
//         }
//     }
// }

// Function to send user message and receive bot response
function sendMessage() {
    const userInput = document.querySelector('.user-input');
    const userMessage = userInput.value.trim();

    if (userMessage !== '') {
        displayMessage('user', userMessage);
        userInput.value = ''; // Clear input field

        // Perform XMLHttpRequest to get bot response
        var request = new XMLHttpRequest();
        var data = "You: " + userMessage + "\n";
        request.open("GET", "http://127.0.0.1:5000/predict?mytext=" + userMessage);
        request.onreadystatechange = function () {
            if (this.readyState === 4 && this.status === 200) {
                var botResponse = JSON.parse(this.responseText)["response"];
                data = data + "Chatbot: " + botResponse + "\n";
                displayMessage('bot', botResponse);
            }
        };
        request.send();
    }
}


// Function to display messages in the chat window
function displayMessage(sender, message) {
    const chatOutput = document.getElementById('chatOutput');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    messageElement.innerText = message;
    chatOutput.appendChild(messageElement);

    // Scroll to bottom
    chatOutput.scrollTop = chatOutput.scrollHeight;
}

