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
$topBarHeight: 80px;

.top_bar {
  height: $topBarHeight;
  display: flex;
  justify-content: space-between;

  box-shadow: #2c2c2c22 0 0 64px;

  .logo_wrapper,
  .nav_wrapper {
    display: inline-flex;
  }

  .logo_wrapper {
    align-items: center;
    padding: 0 40px;

    transition: filter var(--duration);

    &:hover {
      filter: contrast(150%);
    }

    .logo {
      margin-right: 16px;
    }

    .name {
      color: var(--primary);
      margin-top: 8px;
      font-size: 32px;
      font-weight: bold;
    }
  }

  .link {
    color: var(--text);
    padding: 0 25px;
    display: inline-flex;
    font-size: 16px;
    justify-content: center;
    align-items: center;
    transition: background-color var(--duration);

    &:hover {
      background-color: var(--secondary);
      color: #fff;
    }
  }
}
</style>
