document.getElementById("check-btn").addEventListener("click", function () {
    let text = document.getElementById("text-input").value;

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = "Result: " + data.prediction;
    })
    .catch(error => {
        document.getElementById("result").innerText = "Error: " + error;
    });
});
