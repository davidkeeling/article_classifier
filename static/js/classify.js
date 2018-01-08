
$(document).ready(function() {
  $('#classify').on('submit', function (event) {
    event.preventDefault();
    $('#output').slideUp(function () {
      $.ajax({
        url: '/classify',
        method: 'POST',
        data: $(event.target).serialize(),
        success: function (data) {
          $('#output').removeClass('hidden').slideDown();
          $('#output .classification').text(data);
        },
        error: function (jqxhr, status, err) {
          console.log(jqxhr, status, err);
        }
      });
    });
  });
});
