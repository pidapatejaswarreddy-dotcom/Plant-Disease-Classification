(function () {
  "use strict";
  var form = document.getElementById("upload-form");
  if (!form) { return; }
  var input = document.getElementById("image");
  var preview = document.getElementById("preview");
  var previewImg = document.getElementById("preview-img");
  var previewName = document.getElementById("preview-name");
  var submitBtn = document.getElementById("submit-btn");
  var resetBtn = document.getElementById("reset-btn");
  var errBox = document.getElementById("client-error");
  var loading = document.getElementById("loading");
  var zone = document.getElementById("dropzone");
  var maxBytes = parseInt(form.dataset.maxBytes, 10);
  var okTypes = ["image/jpeg", "image/png"];
  var objectUrl = null;

  function showError(msg) { errBox.textContent = msg; errBox.hidden = false; }
  function reset() {
    input.value = ""; preview.hidden = true; errBox.hidden = true; resetBtn.hidden = true;
    submitBtn.disabled = true; loading.hidden = true;
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  }
  function handleFile() {
    errBox.hidden = true;
    var f = input.files && input.files[0];
    if (!f) { reset(); return; }
    var name = f.name.toLowerCase();
    var extOk = /\.(jpe?g|png)$/.test(name);
    if (!extOk || (f.type && okTypes.indexOf(f.type) === -1)) {
      reset(); showError("Unsupported file type. Please choose a JPG, JPEG or PNG image."); return;
    }
    if (f.size > maxBytes) { reset(); showError("The file is too large. Maximum size is " + (maxBytes / 1048576) + " MB."); return; }
    if (objectUrl) { URL.revokeObjectURL(objectUrl); }
    objectUrl = URL.createObjectURL(f);
    previewImg.src = objectUrl; previewName.textContent = f.name;
    preview.hidden = false; resetBtn.hidden = false; submitBtn.disabled = false;
  }
  input.addEventListener("change", handleFile);
  resetBtn.addEventListener("click", reset);
  form.addEventListener("submit", function () {
    submitBtn.disabled = true; loading.hidden = false;
  });
  ["dragenter", "dragover"].forEach(function (ev) {
    zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.add("drag"); });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.remove("drag"); });
  });
  zone.addEventListener("drop", function (e) {
    if (e.dataTransfer && e.dataTransfer.files.length) { input.files = e.dataTransfer.files; handleFile(); }
  });
  window.addEventListener("pageshow", function () { loading.hidden = true; if (input.files && input.files.length) { submitBtn.disabled = false; } });
})();
