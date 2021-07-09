function checkdata()
            {
                var username = document.getElementById("name");
                var emailid = document.getElementById("email");
                if(username.value==""){
                    alert("Please enter the name");
                    username.focus();
                    return false;
                    }
                    if(emailid.value == ""){
                        alert("Please enter the email");
                        emailid.focus();
                        return false;
                  }
                 //If all is well return true.
                  return true;


            }

