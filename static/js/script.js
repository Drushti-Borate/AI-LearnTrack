// static/js/script.js

const subjectsContainer = document.getElementById("subjects-container");
const addSubjectBtn = document.getElementById("add-subject-btn");
const predictBtn = document.getElementById("predict-btn");
const errorMessage = document.getElementById("error-message");

const predictedScoreEl = document.getElementById("predicted-score");
const riskLevelEl = document.getElementById("risk-level");
const learnerTypeEl = document.getElementById("learner-type");
const averageMarksEl = document.getElementById("average-marks");
const suggestionsList = document.getElementById("suggestions-list");

const MAX_SUBJECTS = 5;

let subjectCount = 0;

function createSubjectRow(index) {
  const row = document.createElement("div");
  row.className = "subject-row";
  row.dataset.index = index;

  row.innerHTML = `
    <div class="form-group">
      <label>Subject ${index + 1} Marks</label>
      <input type="number" class="subject-marks" placeholder="e.g. 65">
    </div>
    <div class="form-group">
      <label>Out of</label>
      <input type="number" class="subject-outof" placeholder="e.g. 80">
    </div>
    <button type="button" class="remove-btn" title="Remove subject">&times;</button>
  `;

  const removeBtn = row.querySelector(".remove-btn");
  removeBtn.addEventListener("click", () => {
    subjectsContainer.removeChild(row);
    subjectCount--;
    updateSubjectLabels();
  });

  return row;
}

function updateSubjectLabels() {
  const rows = subjectsContainer.querySelectorAll(".subject-row");
  rows.forEach((row, i) => {
    row.dataset.index = i;
    const label = row.querySelector(".form-group label");
    if (label) {
      label.textContent = `Subject ${i + 1} Marks`;
    }
  });
}

addSubjectBtn.addEventListener("click", () => {
  if (subjectCount >= MAX_SUBJECTS) {
    errorMessage.textContent = "You can add maximum 5 subjects.";
    return;
  }
  errorMessage.textContent = "";
  const row = createSubjectRow(subjectCount);
  subjectsContainer.appendChild(row);
  subjectCount++;
});

// Add 3 subjects by default
for (let i = 0; i < 3; i++) {
  const row = createSubjectRow(subjectCount);
  subjectsContainer.appendChild(row);
  subjectCount++;
}

function setRiskBadge(level) {
  riskLevelEl.className = "badge"; // reset classes
  riskLevelEl.classList.add("badge-neutral");

  const lvl = String(level || "").toLowerCase();
  if (lvl === "low") {
    riskLevelEl.classList.remove("badge-neutral");
    riskLevelEl.classList.add("badge-low");
  } else if (lvl === "medium") {
    riskLevelEl.classList.remove("badge-neutral");
    riskLevelEl.classList.add("badge-medium");
  } else if (lvl === "high") {
    riskLevelEl.classList.remove("badge-neutral");
    riskLevelEl.classList.add("badge-high");
  } else if (lvl === "critical") {
    riskLevelEl.classList.remove("badge-neutral");
    riskLevelEl.classList.add("badge-critical");
  }
}

predictBtn.addEventListener("click", async () => {
  errorMessage.textContent = "";

  const rows = subjectsContainer.querySelectorAll(".subject-row");
  const subjects = [];

  rows.forEach((row) => {
    const marksInput = row.querySelector(".subject-marks");
    const outofInput = row.querySelector(".subject-outof");

    const marksVal = parseFloat(marksInput.value || "0");
    const outofVal = parseFloat(outofInput.value || "0");

    if (!isNaN(marksVal) && !isNaN(outofVal) && outofVal > 0) {
      subjects.push({
        marks: marksVal,
        outof: outofVal,
      });
    }
  });

  if (subjects.length === 0) {
    errorMessage.textContent = "Please enter at least one subject with valid marks and out of.";
    return;
  }

  const previous_year_percentage = parseFloat(
    document.getElementById("previous_year_percentage").value || "0"
  );
  const attendance = parseFloat(
    document.getElementById("attendance").value || "0"
  );
  const study_hours = parseFloat(
    document.getElementById("study_hours").value || "0"
  );
  const sleep_hours = parseFloat(
    document.getElementById("sleep_hours").value || "0"
  );
  const focus_time = parseFloat(
    document.getElementById("focus_time").value || "0"
  );

  const payload = {
    subjects,
    previous_year_percentage,
    attendance,
    study_hours,
    sleep_hours,
    focus_time,
  };

  try {
    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";

    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    // Update UI
    predictedScoreEl.textContent = data.predicted_score ?? "--";
    riskLevelEl.textContent = data.risk_level ?? "--";
    learnerTypeEl.textContent = data.learner_type ?? "--";
    averageMarksEl.textContent = data.average_marks_percent ?? "--";

    setRiskBadge(data.risk_level);

    // Suggestions
    suggestionsList.innerHTML = "";
    if (Array.isArray(data.suggestions) && data.suggestions.length > 0) {
      data.suggestions.forEach((sugg) => {
        const li = document.createElement("li");
        li.textContent = sugg;
        suggestionsList.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "No suggestions generated.";
      li.classList.add("placeholder");
      suggestionsList.appendChild(li);
    }
  } catch (err) {
    console.error(err);
    errorMessage.textContent = err.message;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict Performance";
  }
});
