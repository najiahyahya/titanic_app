const form = document.getElementById("predict-form");
const resultCard = document.getElementById("resultCard");
const resultText = document.getElementById("resultText");
const rawJson = document.getElementById("rawJson");
const meterFill = document.getElementById("meterFill");
const randomBtn = document.getElementById("randomBtn");

// --- Hybrid input sync ---
function syncHybrid(rangeId, numberId) {
  const range = document.getElementById(rangeId);
  const number = document.getElementById(numberId);

  // Update number when range changes
  range.addEventListener("input", () => {
    number.value = range.value;
  });

  // Update range when number changes
  number.addEventListener("input", () => {
    range.value = number.value;
  });
}

// Apply sync to Age + Fare
syncHybrid("Age", "AgeNumber");
syncHybrid("Fare", "FareNumber");

// --- Random Example button ---
randomBtn.addEventListener("click", () => {
  const rnd = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
  document.getElementById("Pclass").value = [1, 2, 3][rnd(0, 2)];

  const sexes = document.querySelectorAll("input[name='Sex']");
  sexes[rnd(0, 1)].checked = true;

  const rndFloat = (min, max, decimals = 2) => {
    const factor = Math.pow(10, decimals);
    return (Math.random() * (max - min) + min).toFixed(decimals);
  };
  
  const age = rnd(1, 80);
  document.getElementById("Age").value = age;
  document.getElementById("AgeNumber").value = age;

  const fare = rnd(0, 512.33);
  document.getElementById("Fare").value = fare;
  document.getElementById("FareNumber").value = fare;

  document.getElementById("SibSp").value = rnd(0, 5);
  document.getElementById("Parch").value = rnd(0, 4);
  document.getElementById("Embarked").value = ["S", "C", "Q"][rnd(0, 2)];
});

// --- Form Submit (Predict) ---
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = new FormData(form);
  const payload = Object.fromEntries(data.entries());

  // Radios: ensure string not array
  payload.Sex = document.querySelector("input[name='Sex']:checked").value;

  // Show "loading" state
  resultCard.classList.remove("hidden");
  resultText.textContent = "Predicting…";
  meterFill.style.width = "0%";
  rawJson.textContent = "";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const json = await res.json();
    if (!json.ok) {
      resultText.textContent = "Error: " + (json.error || "Unknown error");
      rawJson.textContent = JSON.stringify(json, null, 2);
      return;
    }

    const pct = Math.round(json.probability * 100);
    resultText.innerHTML = `<strong>${json.prediction}</strong> — Confidence: <strong>${pct}%</strong>`;
    meterFill.style.width = pct + "%";
    rawJson.textContent = JSON.stringify(json, null, 2);
  } catch (err) {
    resultText.textContent = "Network or server error: " + err.message;
  }
});