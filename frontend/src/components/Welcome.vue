<template>
  <div class="welcome_wrapper">
    <div class="welcome">
      <section class="intro_section">
        <h1 class="title">Facial Image Compression Tool</h1>
        <div class="intro">
          This is an online compression tool specially optimized for facial images based on deep
          learning.
        </div>
        <div class="btn_container">
          <router-link class="try_btn" to="compression">Try now!</router-link>
          <a class="more_btn" href="#">learn more</a>
        </div>
      </section>
      <img class="logo" src="../assets/logo.png" alt="logo" />
    </div>
  </div>
</template>

<script>
export default {};
</script>

<style lang="scss" scoped>
.welcome_wrapper {
  height: 100vh;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.welcome {
  height: 250px;
  width: 900px;
  margin: 0 auto;
  display: flex;

  .intro_section {
    max-width: 550px;
    margin-right: 96px;

    .title {
      font-weight: bold;
      font-size: 48px;
      margin: 0 0 18px;
    }

    .intro {
      font-size: 20px;
    }
  }

  .btn_container {
    margin-top: 24px;

    .try_btn,
    .more_btn {
      display: inline-block;
      height: 50px;
      width: 140px;
      border-radius: 25px;
      text-align: center;
      font-size: 20px;
      padding: 16px;
    }

    .try_btn {
      border: none;
      background-color: #0052d9;
      color: #fff;
      margin-right: 16px;
    }

    .more_btn {
      border: 2px #0052d9 solid;
      background-color: #fff;
      color: #0052d9;
      padding: 14px;
    }
  }

  .logo {
    display: block;
    height: 250px;
    width: 250px;
  }
}
</style>
