document.getElementById("check-btn").addEventListener("click", async () => {
    const text = document.getElementById("text-input").value;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        const result = await response.json();
        if (result.error) {
            document.getElementById("result").innerText = "Error: " + result.error;
        } else {
            document.getElementById("result").innerText = "Result: " + result.prediction;
        }

    } catch (err) {
        document.getElementById("result").innerText = "Error: " + err;
    }
});
