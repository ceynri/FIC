<template>
  <div class="top_bar">
    <div class="logo_wrapper">
      <img
        class="logo"
        src="../assets/logo.png"
        alt="logo"
      />
      <span class="name">FIC</span>
    </div>
    <div class="right_wrapper">
      <router-link to="about" class="link about">About</router-link>
      <a class="link github" href="http://github.com/ceynri/FIC">Github</a>
    </div>
  </div>
</template>

<script>
export default {};
</script>

<style lang="scss" scoped>
.top_bar {
  height: 80px;
  display: flex;
  justify-content: space-between;

  box-shadow: #2c2c2c22 0 0 64px;

  .logo_wrapper,
  .right_wrapper {
    display: inline-flex;
  }

  .logo_wrapper {
    align-items: center;
    padding: 0 40px;

    .logo {
      height: 32px;
      margin-right: 16px;
    }

    .name {
      color: #0052d9;
      margin-top: 8px;
      font-size: 32px;
      font-weight: bold;
    }
  }

  .link {
    color: #2c2c2c;
    width: 120px;
    display: inline-flex;
    font-size: 20px;
    justify-content: center;
    align-items: center;

    &:hover {
      background-color: #0052d9;
      color: #fff;
    }
  }
}
</style>
