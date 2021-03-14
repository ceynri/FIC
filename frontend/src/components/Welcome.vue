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
          <router-link class="try_btn btn" to="demo">Try now!</router-link>
          <router-link class="more_btn btn" to="compress">Compress</router-link>
        </div>
      </section>
      <IconBase class="logo" icon-name="logo" width="250" height="250">
        <Favicon />
      </IconBase>
    </div>
  </div>
</template>

<script>
import Favicon from '@/components/icons/Favicon.vue';

export default {
  components: {
    Favicon,
  },
};
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
      color: var(--text);
      font-weight: bold;
      font-size: 48px;
      margin: 0 0 18px;
    }

    .intro {
      color: var(--text2);
      font-size: 20px;
      line-height: 1.5;
    }
  }

  .btn_container {
    margin-top: 26px;

    .btn {
      display: inline-block;
      height: 50px;
      width: 140px;
      border-radius: 25px;
      text-align: center;
      white-space: nowrap;
      font-size: 18px;
      padding: 16px;

      transition: filter var(--duration), box-shadow var(--duration);

      &:hover {
        filter: contrast(150%) drop-shadow(2px 4px 16px var(--shadow));
      }
    }

    .try_btn {
      border: none;
      background-color: var(--primary);
      color: #fff;
      margin-right: 16px;
    }

    .more_btn {
      border: 2px var(--primary) solid;
      background-color: initial;
      color: var(--primary);
      padding: 14px;
    }
  }

  .logo {
    display: block;
    height: 250px;
    width: 250px;
    color: var(--secordary);
  }
}
</style>
